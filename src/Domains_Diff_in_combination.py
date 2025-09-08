import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
from src import Evaluate
from src import Models
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score
from pathlib import Path

def model_fit_and_store_result(df, model_params, group_number, domain_lst, domains_to_iter, model_eval):
    # model fit
    #model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst_without_goal, domains_to_iter_without_goal, model_eval)
    model = Models.Model_fixed_test_size(data=df, model_params=model_params, domain=domain_lst, model='lgb', train_subset_size=1, order=0)


    pred_prob, pred_label = model.test_set_predict_prob, model.test_set_predict

    if model.samp_weight_control:
        y_test, sample_weight = model.y_test, model.test_sample_weight

        # pr part
        precision, recall, _ = precision_recall_curve(y_test, pred_prob, sample_weight=sample_weight)
        pr_f1, pr_auc = f1_score(y_test, pred_label, sample_weight=sample_weight), auc(recall, precision)
        pr_no_skill = len(y_test[y_test == 1]) / len(y_test)

        # roc
        roc_no_skill = 0.5
        auc_score = roc_auc_score(y_test, pred_prob, sample_weight=sample_weight)

        # brier
        brier = Evaluate.brier(y_test, pred_prob)

        # imv
        imv = Evaluate.imv(y_test, model.y_train, pred_prob)
    else:
        evaluate=Evaluate.metric(model=model)

    '''    # domain name
        domain_name, switch = '', True
        for domain in domains_to_iter:
            if domain == 'demographic':
                continue
            else:
                if switch:
                    domain_name += domain + '+'
                    switch = False
                else:
                    domain_name += domain + '\n'
                    switch = True
        if domain_name.endswith('+'): domain_name = domain_name[0:len(domain_name) - 1]
        if domain_name == '':
            domain_name = 'demographic'
            domains_to_iter = ['demographic']'''
    # columns=['model','goal_domain','group_number', 'domain_list', 'domain_num', 'domain_names',
    #              'pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc']

    # print('domains to store:{}'.format(domains_to_iter))
    model_eval.loc[len(model_eval)] = ['lgb', group_number, domains_to_iter, len(domains_to_iter),evaluate.pr_no_skill, evaluate.pr_f1,
                                       evaluate.pr_auc, 0.5, evaluate.auc_score, evaluate.brier, evaluate.imv]
    return model_eval


def get_dc_params(domain_name_lst):
    domain_name_lst=list(domain_name_lst)
    for remove_domain  in ['all','all_bio','all_bio_adjusted','bio']:
        if remove_domain in domain_name_lst: domain_name_lst.remove(remove_domain)

    df_eval = pd.DataFrame(columns=['model', 'group_number', 'domain_list', 'domain_num',
                                      'pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc', 'brier', 'imv'])
    iterations = []
    for iterate_num in np.arange(1, len(domain_name_lst)+1, 1):
        iterations += list(itertools.combinations(domain_name_lst, iterate_num))


    return domain_name_lst,df_eval,iterations

def dc_iteration(iterations,df_eval,df,model_params):
    domains = model_params['domain_dict']
    for domains_to_iter in iterations:

        group_number = int(iterations.index(domains_to_iter))
        domains_to_iter = list(domains_to_iter)

        domain_lst = domains['demographic'] if 'demographic' in domains_to_iter else []

        for single_domain in domains_to_iter: domain_lst = list(set(domain_lst + domains[single_domain]))
        domain_lst = [x for x in domain_lst if x in list(df.columns)]
        df_eval = model_fit_and_store_result(df, model_params, group_number, domain_lst, domains_to_iter, df_eval)
    return df_eval


# post process-------------------------------------------------------------
def df_post_process(domain_name_lst,df_eval):
    model_eval_diff = pd.DataFrame(
        columns=['goal_domain', 'group_number', 'domain_list', 'domain_num', 'pr_f1', 'pr_auc', 'roc_auc', 'brier', 'imv'])
    contribution = pd.DataFrame(
        columns=['goal_domain', 'pr_f1_contribution', 'pr_auc_contribution', 'roc_auc_contribution', 'brier_contribution', 'imv_contribution'])

    for goal_domain in domain_name_lst:
        post_process_model_eval = df_eval.loc[df_eval['goal_domain'] == goal_domain]
        post_groups = list(set(post_process_model_eval['group_number']))

        for group_number in post_groups:
            post_to_process = post_process_model_eval[post_process_model_eval['group_number'] == group_number]

            post_row_min = post_to_process[post_to_process['domain_num'] == min(post_to_process['domain_num'])].to_dict(
                'list')

            post_row_max = post_to_process[post_to_process['domain_num'] == max(post_to_process['domain_num'])].to_dict(
                'list')

            # columns=['goal_domain','group_number','domain_list','domain_num','pr_f1','pr_auc','roc_auc','brier','imv']
            model_eval_diff.loc[len(model_eval_diff)] = [post_row_max['goal_domain'][0], post_row_max['group_number'][0],
                                                         post_row_max['domain_list'][0], post_row_max['domain_num'][0],
                                                         post_row_max['pr_f1'][0] - post_row_min['pr_f1'][0],
                                                         post_row_max['pr_auc'][0] - post_row_min['pr_auc'][0],
                                                         post_row_max['roc_auc'][0] - post_row_min['roc_auc'][0],
                                                         post_row_max['brier'][0] - post_row_min['brier'][0],
                                                         post_row_max['imv'][0] - post_row_min['imv'][0]]

        pr_f1_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_f1'].mean()
        pr_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_auc'].mean()
        roc_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'roc_auc'].mean()
        brier_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'brier'].mean()
        imv_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'imv'].mean()
        # columns=['goal_domain','pr_f1_contribution','pr_auc_contribution','roc_auc_contribution','brier_contribution','imv_contribution']
        contribution.loc[len(contribution)] = [goal_domain, pr_f1_contribution, pr_auc_contribution, roc_auc_contribution, brier_contribution, imv_contribution]
    return contribution, model_eval_diff

# contribution_xgb = contribution.copy()

def get_target_domain_iterations(goal_domain,iterations):
    return [x for x in iterations if goal_domain in x ]
def dd_post_process(domain_name_lst,iterations,df_eval):
    iterations = [set(x) for x in iterations]
    # this is the version without requiring demographic as a fundamental domain
    model_eval_diff = pd.DataFrame(columns=['goal_domain', 'group_number', 'domain_list', 'domain_num', 'pr_f1', 'pr_auc', 'roc_auc', 'brier', 'imv'])
    contribution = pd.DataFrame(columns=['goal_domain', 'pr_f1_contribution', 'pr_auc_contribution', 'roc_auc_contribution', 'brier_contribution', 'imv_contribution'])


    for goal_domain in domain_name_lst:
        goal_comb_lst = get_target_domain_iterations(goal_domain,iterations)
        for combination in goal_comb_lst:
            if len(combination)>1:
                # get the group number first
                group_number = int(iterations.index(combination))

                combination_without = combination-{goal_domain}
                group_number_without_goal = int(iterations.index(combination_without))

                row = df_eval.loc[df_eval['group_number']==group_number]
                row_without = df_eval.loc[df_eval['group_number']==group_number_without_goal]


                # store the results
                model_eval_diff.loc[len(model_eval_diff)] = [goal_domain, row['group_number'].values[0],
                                                             row['domain_list'].values[0], row['domain_num'].values[0],
                                                             row['pr_f1'].values[0] - row_without['pr_f1'].values[0],
                                                             row['pr_auc'].values[0] - row_without['pr_auc'].values[0],
                                                             row['roc_auc'].values[0] - row_without['roc_auc'].values[0],
                                                             row['brier'].values[0] - row_without['brier'].values[0],
                                                             row['imv'].values[0] - row_without['imv'].values[0]]
            else:
                group_number = int(iterations.index(combination))
                row = df_eval.loc[df_eval['group_number']==group_number]

                model_eval_diff.loc[len(model_eval_diff)] = [goal_domain, row['group_number'].values[0],
                                                             row['domain_list'].values[0], 1,
                                                             row['pr_f1'].values[0] ,
                                                             row['pr_auc'].values[0] ,
                                                             row['roc_auc'].values[0],
                                                             row['brier'].values[0],
                                                             row['imv'].values[0]]


        pr_f1_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_f1'].mean()
        pr_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_auc'].mean()
        roc_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'roc_auc'].mean()
        brier_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'brier'].mean()
        imv_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'imv'].mean()
        contribution.loc[len(contribution)] = [goal_domain, pr_f1_contribution, pr_auc_contribution, roc_auc_contribution, brier_contribution, imv_contribution]
    return model_eval_diff, contribution

#plot part
def dc_plot(contribution,save_control=False):
    color_yellow='#F1A52C'
    color_blue='#001C5B'
    replace_name_dict={'demographic':'Demography',
                       'child_adverse':'Child-Adversity',
                      'adult_SES':'Socioeconomic',
                      'behavioral':'Behaviours',
                      'adult_adverse':'Adversity',
                      'social_connection':'Connections',
                      'psych':'Psychological ',
                      'bio_adjusted':'Biomarkers',
                       'pr_f1_contribution':'F1 Score',
                       'pr_auc_contribution':'PR-AUC Score',
                      'roc_auc_contribution':'ROC-AUC Score',
                      'imv_contribution':'IMV'}



    #contribution.to_csv(Path.cwd()/'results/domain_contribution_lgb.csv')

    #contribution_xgb.to_csv(Path.cwd()/'results/domain_contribution_xgb.csv')


    #contribution=pd.read_csv(Path.cwd()/'results/domain_contribution_lgb.csv',index_col=0)

    figure, axis = plt.subplots(2, 2)
    fontsize_ticks = 18
    fontsize_labels = 20
    plt.rcParams["figure.figsize"] = [20,14]
    figure.subplots_adjust(left=0.12, bottom=0.1,top=0.995,right=0.995)

    count=0
    ploted_column = []
    for (m, n), subplot in np.ndenumerate(axis):
        column=list(contribution.columns)[count]

        while (column not in replace_name_dict.keys()) | (column in ploted_column):
            count+=1
            column = list(contribution.columns)[count]

        print(column)
        count += 1
        ploted_column.append(column)
        axis[m,n].barh([replace_name_dict[x_tick] for x_tick in contribution['goal_domain']],contribution[column],color=color_blue,alpha=0.75)
        axis[m,n].set_xlabel(replace_name_dict[column],size=fontsize_labels)
        axis[m, n].set_yticklabels([replace_name_dict[x_tick] for x_tick in contribution['goal_domain']])
        axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        axis[m, n].spines['top'].set_visible(False)
        axis[m, n].spines['right'].set_visible(False)
        axis[m,n].grid(axis='x',alpha=0.4)
        axis[m, n].set_axisbelow(True)
        if m==1:
            axis[m, n].locator_params(axis='x', nbins=7)
    if save_control:
        plt.savefig(Path.cwd() / 'graphs/domain_contribution_lgbm.pdf')
    figure.tight_layout()
    plt.show()







