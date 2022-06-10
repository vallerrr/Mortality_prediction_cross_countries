import numpy as np
import pandas as pd
import itertools
from src import DataImport
from src import Evaluate
from src import Models
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score


def model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst, domains_to_iter, model_eval):
    # model fit
    model = Models.Model_fixed_test_size(data=df, test_size=test_size, domain_list=domain_lst, model='xgb',
                                  train_subset_size=1, order=0)
    pred_prob, pred_label = model.test_set_predict_prob, model.test_set_predict
    y_test, sample_weight = model.y_test, model.test_sample_weight

    # pr part
    precision, recall, _ = precision_recall_curve(y_test, pred_prob, sample_weight=sample_weight)
    pr_f1, pr_auc = f1_score(y_test, pred_label, sample_weight=sample_weight), auc(recall, precision)
    pr_no_skill = len(y_test[y_test == 1]) / len(y_test)

    # roc
    roc_no_skill = 0.5
    auc_score = roc_auc_score(y_test, pred_prob, sample_weight=sample_weight)

    #brier
    brier = Evaluate.brier(y_test, pred_prob)

    # domain name
    domain_name, switch = '', True
    for domain in domains_to_iter:
        if domain == 'demographic':
            # continue
            if switch:
                domain_name += domain + '+'
                switch = False
            else:
                domain_name += domain + '\n'
                switch = True
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
                                       domain_name, pr_no_skill, pr_f1,
                                       pr_auc,
                                       roc_no_skill, auc_score, brier]

    return model_eval


# Bio_data Import and Initialization -----------------------------------------
test_size = 0.3
df = DataImport.data_reader()
domains = DataImport.domain_dict()
domain_name_lst = list(domains.keys())
domain_name_lst.remove('all')
# domain_name_lst.remove('demographic')
# data structure to store info
model_eval = pd.DataFrame(columns=['model', 'goal_domain', 'group_number',
                                   'domain_list', 'domain_num', 'domain_names',
                                   'pr_no_skill', 'pr_f1', 'pr_auc',
                                   'roc_no_skill', 'roc_auc', 'brier'])
# -------------------------------------------------------------------------


# Create combinations to iterate ------------------------------------------
iterations = []
for iterate_num in np.arange(2, 8, 1):
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
            #domain_lst_without_goal = domains['demographic']
            domain_lst_without_goal = []
            for single_domain in domains_to_iter_without_goal: domain_lst_without_goal = list(
                set(domain_lst_without_goal + domains[single_domain]))
            model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst_without_goal,
                                                    domains_to_iter_without_goal, model_eval)

            # result with the domain
            # domain_lst = domains['demographic']
            domain_lst = []
            for single_domain in domains_to_iter: domain_lst = list(set(domain_lst + domains[single_domain]))
            model_eval = model_fit_and_store_result(df, goal_domain, group_number, test_size, domain_lst,
                                                    domains_to_iter, model_eval)

        # model_eval.loc[model_eval['domain_names'] == 'demographic', 'domain_num'] = 0

# -------------------------------------------------------------------------

# post process-------------------------------------------------------------
model_eval_diff = pd.DataFrame(
    columns=['goal_domain', 'group_number', 'domain_list', 'domain_num', 'pr_f1', 'pr_auc', 'roc_auc','brier'])
model_eval_diff_percent = pd.DataFrame(
    columns=['goal_domain', 'group_number', 'domain_list', 'domain_num', 'pr_f1', 'pr_auc', 'roc_auc','brier'])
contribution = pd.DataFrame(
    columns=['goal_domain',
             'f1_contribution','f1_percent',
             'pr_auc_contribution','pr_auc_percent',
             'roc_auc_contribution','roc_auc_percent',
             'brier_contribution','brier_percent'])

for goal_domain in domain_name_lst:
    post_process_model_eval = model_eval.loc[model_eval['goal_domain'] == goal_domain]
    psot_groups = list(set(post_process_model_eval['group_number']))

    for group_number in psot_groups:
        post_to_process = post_process_model_eval[post_process_model_eval['group_number'] == group_number]

        post_row_min = post_to_process[post_to_process['domain_num'] == min(post_to_process['domain_num'])].to_dict(
            'list')

        post_row_max = post_to_process[post_to_process['domain_num'] == max(post_to_process['domain_num'])].to_dict(
            'list')

        # columns=['goal_domain','group_number','domain_list','domain_num','pr_f1','pr_auc','roc_auc','brier']
        model_eval_diff.loc[len(model_eval_diff)] = [post_row_max['goal_domain'][0], post_row_max['group_number'][0],
                                                     post_row_max['domain_list'][0], post_row_max['domain_num'][0], \
                                                     post_row_max['pr_f1'][0] - post_row_min['pr_f1'][0],
                                                     post_row_max['pr_auc'][0] - post_row_min['pr_auc'][0],
                                                     post_row_max['roc_auc'][0] - post_row_min['roc_auc'][0],
                                                     post_row_max['brier'][0] - post_row_min['brier'][0]]
        # store difference of relative percentage
        model_eval_diff_percent.loc[len(model_eval_diff_percent)] = [post_row_max['goal_domain'][0], post_row_max['group_number'][0],
                                                                     post_row_max['domain_list'][0], post_row_max['domain_num'][0],
                                                                     (post_row_max['pr_f1'][0] - post_row_min['pr_f1'][0])/post_row_max['pr_f1'][0],
                                                                     (post_row_max['pr_auc'][0] - post_row_min['pr_auc'][0])/post_row_max['pr_auc'][0] ,
                                                                     (post_row_max['roc_auc'][0] - post_row_min['roc_auc'][0])/post_row_max['roc_auc'][0],
                                                                     (post_row_max['brier'][0] - post_row_min['brier'][0])/post_row_max['brier'][0]]

    f1_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_f1'].mean()
    pr_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'pr_auc'].mean()
    roc_auc_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'roc_auc'].mean()
    brier_contribution = model_eval_diff.loc[model_eval_diff['goal_domain'] == goal_domain, 'brier'].mean()

    f1_contribution_percent = model_eval_diff_percent.loc[model_eval_diff_percent['goal_domain'] == goal_domain, 'pr_f1'].mean()
    pr_auc_contribution_percent = model_eval_diff_percent.loc[model_eval_diff_percent['goal_domain'] == goal_domain, 'pr_auc'].mean()
    roc_auc_contribution_percent = model_eval_diff_percent.loc[model_eval_diff_percent['goal_domain'] == goal_domain, 'roc_auc'].mean()
    brier_contribution_percent = model_eval_diff_percent.loc[model_eval_diff_percent['goal_domain'] == goal_domain, 'brier'].mean()

    # columns=['goal_domain',
    #              'f1_contribution','f1_percent',
    #              'pr_auc_contribution','pr_auc_percent',
    #              'roc_auc_contribution','roc_auc_percent',
    #              'brier_contribution','brier_percent']

    contribution.loc[len(contribution)] = [goal_domain, f1_contribution, f1_contribution_percent,
                                           pr_auc_contribution, pr_auc_contribution_percent,
                                           roc_auc_contribution, roc_auc_contribution_percent,
                                           brier_contribution, brier_contribution_percent]


