"""
# Created by valler at 22/01/2025
Feature: 

"""
from src import params
import numpy as np
import pandas as pd
from pathlib import Path
from src.Domains_Diff_in_combination import get_dc_params,dc_iteration,dd_post_process,dc_plot
import warnings
import os
import random
warnings.filterwarnings("ignore")
random.seed(87785)
np.random.seed(87785)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'



contribution = pd.DataFrame()
basic_domains = ['demographic', 'child_adverse', 'adult_SES', 'behavioral', 'adult_adverse', 'social_connection', 'psych']
colors = {'blue':'#62a0cb','orange':'#ff7f0e','green':'#2ca02c'}

for dataset in ['HRS','SHARE','ELSA','COMB']:
    print(dataset)
    domain_name = dataset
    model_params = params.model_params

    if dataset == 'HRS':
        df = pd.read_pickle('/Users/valler/Python/OX_Thesis/OX_thesis/data/HRS/data_preprocess/Data/merge_data_not_standardise_no_missing.pkl')
    elif dataset == 'COMB':
        comb_type = 3
        domain_name = 'combination_all'
        df, model_params['domain_dict'][domain_name] = params.read_merged_data(type=comb_type)
        model_params['domain_dict'][domain_name].sort()
        model_params['var_dict']['dataset'] = 'Datasource'
    else:
        df = params.data_reader(dataset=dataset, source='us', bio=False)

    if dataset !='COMB':
        model_params['domain_dict'][domain_name]= list(set(model_params['domain_dict']['all']).intersection(set(df.columns)))
        model_params['domain_dict'][domain_name].sort()

    domains = []
    for var in model_params['domain_dict'][domain_name]:
        for domain in basic_domains:
            if (var in model_params['domain_dict'][domain] )& (var not in model_params['domain_dict']['demographic']) :
                domains+=[domain]
    domains = list(set(domains))+['demographic']
    domains.sort()

    domain_name_lst,df_eval,iterations = get_dc_params(domains)

    df_eval = dc_iteration(iterations,df_eval,df,model_params)
    model_eval_diff_hrs,contribution_hrs = dd_post_process(domain_name_lst,iterations,df_eval)
    contribution_hrs['dataset']=dataset
    dc_plot(contribution_hrs,save_control=False)

    contribution = pd.concat([contribution, contribution_hrs], axis=0)
    del df,model_params


contribution.to_csv(Path.cwd()/'results/domain_contribution_all_lgb_20250122.csv')


# plot the contribution
import matplotlib.pyplot as plt
import numpy as np

orders = {'adult_adverse': 6, 'adult_SES': 3, 'social_connection': 1, 'psych': 0, 'demographic': 5, 'behavioral': 4, 'child_adverse': 2}
replace_name_dict = {'demographic': 'Demography',
                     'child_adverse': 'Child-Adversity',
                     'adult_SES': 'Socioeconomic',
                     'behavioral': 'Behaviours',
                     'adult_adverse': 'Adversity',
                     'social_connection': 'Connections',
                     'psych': 'Psychological ',
                     'bio_adjusted': 'Biomarkers',
                     'pr_f1_contribution': 'F1 Score',
                     'pr_auc_contribution': 'PR-AUC Score',
                     'roc_auc_contribution': 'ROC-AUC Score',
                     'imv_contribution': 'IMV'}
figure = plt.figure()
fontsize_ticks = 14
fontsize_labels = 16
plt.rcParams["figure.figsize"] = [13, 12]
figure.subplots_adjust(left=0.12, bottom=0.1, top=0.995, right=0.995)
datasets = ['HRS','SHARE','ELSA','COMB']

for m in range(1, 5):
    df = contribution.loc[contribution['dataset'] == datasets[m - 1]].copy()
    df['order'] = [orders[x] for x in df['goal_domain']]

    df.sort_values(by=['order'], inplace=True)

    # df.drop(columns=['order'],inplace=True)

    eva_score = 'pr_auc'

    values = list(df[f'{eva_score}_contribution'])
    values += [values[0]]

    labels = [replace_name_dict[x] for x in df['goal_domain']]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    angles = [x + 0.1 for x in angles]  # Close the heptagon

    axis = figure.add_subplot(int(f'22{m}'), polar=True)

    # Plot the heptagon shape
    axis.plot(angles, values, color='black', linewidth=1.2, linestyle='solid')
    axis.fill(angles, values, color=colors['blue'])

    axis.set_ylim([0, max(contribution[f'{eva_score}_contribution'])])

    if datasets[m - 1] == 'ELSA':
        print(values)
        axis.set_ylim([-0.01, max(contribution[f'{eva_score}_contribution'])])

    # Set the labels for each dimension
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([])

    axis.spines['polar'].set_visible(False)
    # ax.spines['polar'].set_alpha(0.1)

    # set title
    if datasets[m - 1] == 'comb':
        axis.text(1.67, 0.25, "HRS + ELSA", ha='center', va='center', fontsize=fontsize_labels, weight="bold")
    else:
        axis.text(1.67, 0.25, datasets[m - 1].upper(), ha='center', va='center', fontsize=fontsize_labels, weight="bold")

    # Add the dimension labels
    for i, angle in enumerate(angles[:-1]):
        x = angle
        y = 0.220
        axis.text(x, y, f'{labels[i]}\n{round(values[i], 3)}', ha='center', va='center', fontsize=fontsize_ticks)

        axis.grid(True, linestyle='dashed', linewidth=1, alpha=0.5, which='major')
        axis.set_yticklabels([])

    # Set the aspect ratio to equal
    axis.set_aspect('equal')
    axis.set_axisbelow(True)
figure.tight_layout()
# Show the plot
plt.show()
#plt.savefig(Path.cwd() / 'graphs/model_outputs/domain_contribution_all.pdf')

