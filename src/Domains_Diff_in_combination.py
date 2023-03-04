import numpy as np
import pandas as pd
import itertools
from src import DataImport
import matplotlib.pyplot as plt
from src import Evaluate
from src import Models
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score
from pathlib import Path

def model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst, domains_to_iter, model_eval):
    # model fit
    #model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst_without_goal, domains_to_iter_without_goal, model_eval)
    model = Models.Model_fixed_test_size(data=df, test_size=test_size, domain_list=domain_lst, model='lgb',
                                  train_subset_size=1, order=0,y_colname='death')
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

    # domain name
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
        domains_to_iter = ['demographic']
    # columns=['model','goal_domain','group_number', 'domain_list', 'domain_num', 'domain_names',
    #              'pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc']

    # print('domains to store:{}'.format(domains_to_iter))
    model_eval.loc[len(model_eval)] = [model, goal_domain, group_number, domains_to_iter, len(domains_to_iter),
                                       domain_name, evaluate.pr_no_skill ,evaluate.pr_f1 ,
                                       evaluate.pr_auc , 0.5, evaluate.auc_score, evaluate.brier, evaluate.imv]

    return model_eval


# model_used_data Import and Initialization -----------------------------------------
test_size = 0.3
df = DataImport.data_reader_by_us(bio=True)
domains = DataImport.domain_dict()
domain_name_lst = list(domains.keys())
domain_name_lst.remove('all')
domain_name_lst.remove('all_bio')
domain_name_lst.remove('all_bio_adjusted')
domain_name_lst.remove('bio')
# data structure to store info
model_eval = pd.DataFrame(columns=['model', 'goal_domain', 'group_number', 'domain_list', 'domain_num', 'domain_names',
                                   'pr_no_skill', 'pr_f1', 'pr_auc', 'roc_no_skill', 'roc_auc', 'brier', 'imv'])
# -------------------------------------------------------------------------


# Create combinations to iterate ------------------------------------------
iterations = []
for iterate_num in np.arange(1, len(domain_name_lst)+1, 1):
    iterations += list(itertools.combinations(domain_name_lst, iterate_num))

# -------------------------------------------------------------------------


# Calculation---------------------------------------------------------------
for goal_domain in domain_name_lst:
    print('working on domain {}'.format(goal_domain))
    for domains_to_iter in iterations:
        if goal_domain in domains_to_iter:
            group_number = int(iterations.index(domains_to_iter))
            # get variable names
            domains_to_iter = list(domains_to_iter)
            domains_to_iter_without_goal = domains_to_iter.copy()

            # without the goal domain
            domains_to_iter_without_goal.remove(goal_domain)
            domain_lst_without_goal = domains['demographic']
            for single_domain in domains_to_iter_without_goal:
                domain_lst_without_goal = list(
                set(domain_lst_without_goal + domains[single_domain]))

            model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst_without_goal,
                                                    domains_to_iter_without_goal, model_eval)

            # result with the domain
            domain_lst = domains['demographic']
            for single_domain in domains_to_iter: domain_lst = list(set(domain_lst + domains[single_domain]))
            model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst,
                                                    domains_to_iter, model_eval)

        model_eval.loc[model_eval['domain_names'] == 'demographic', 'domain_num'] = 0

# -------------------------------------------------------------------------

# post process-------------------------------------------------------------
model_eval_diff = pd.DataFrame(
    columns=['goal_domain', 'group_number', 'domain_list', 'domain_num', 'pr_f1', 'pr_auc', 'roc_auc','brier','imv'])
contribution = pd.DataFrame(
    columns=['goal_domain', 'pr_f1_contribution', 'pr_auc_contribution', 'roc_auc_contribution','brier_contribution','imv_contribution'])


for goal_domain in domain_name_lst:
    post_process_model_eval = model_eval.loc[model_eval['goal_domain'] == goal_domain]
    psot_groups = list(set(post_process_model_eval['group_number']))

    for group_number in psot_groups:
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
    contribution.loc[len(contribution)] = [goal_domain, pr_f1_contribution, pr_auc_contribution, roc_auc_contribution, brier_contribution,imv_contribution]

# contribution_xgb = contribution.copy()


#plot part
color_yellow='#F1A52C'
color_blue='#001C5B'
replace_name_dict={'child_adverse':'Child-Adversity',
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


contribution=pd.read_csv(Path.cwd()/'results/domain_contribution_lgb.csv',index_col=0)
contribution_to_plt=contribution.copy()


figure, axis = plt.subplots(2, 2)
fontsize_ticks = 18
fontsize_labels = 20
plt.rcParams["figure.figsize"] = [17,16]
figure.subplots_adjust(left=0.12, bottom=0.1,top=0.995,right=0.995)

count=0
ploted_column = []
for (m, n), subplot in np.ndenumerate(axis):
    column=list(contribution_to_plt.columns)[count]

    while (column not in replace_name_dict.keys()) | (column in ploted_column):
        count+=1
        column = list(contribution_to_plt.columns)[count]

    print(column)
    count += 1
    ploted_column.append(column)
    axis[m,n].barh([replace_name_dict[x_tick] for x_tick in contribution_to_plt['goal_domain']],contribution_to_plt[column],color=color_blue,alpha=0.75)
    axis[m,n].set_xlabel(replace_name_dict[column],size=fontsize_labels)
    axis[m, n].set_yticklabels([replace_name_dict[x_tick] for x_tick in contribution_to_plt['goal_domain']])
    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m,n].grid(axis='x',alpha=0.4)
    axis[m, n].set_axisbelow(True)
    if m==1:
        axis[m, n].locator_params(axis='x', nbins=7)

plt.show()




plt.savefig(Path.cwd()/'graphs/domain_contribution_lgbm.pdf')
