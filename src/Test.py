import os
import pathlib
import sys

sys.path.append("/gpfs3/users/mills/qlr082/OX_thesis/")

from src import params
import numpy as np
import pandas as pd
from src.Evaluate import sl_only_eva
from src import SuperLearner
import random

platform = "other"
params.confirm_cwd(platform)


# specification
model_params = params.model_params
model_params['k'] = 5
model_params['y_colname'] = 'death'
domain_name = 'combination_all'
df = pd.read_csv('/well/mills/users/qlr082/combined_data.csv')
model_params['domain_dict'][domain_name] = ['Zpessimism', 'everunemployed', 'Zperceivedconstraints', 'Zmotherseduc', 'rocc', 'nevermarried', 'ZwealthT', 'age', 'everrent', 'dataset', 'Zoptimism', 'maleYN', 'Znegaffect', 'sleepYN', 'migrantYN', 'fathersocc', 'Zfatherseduc', 'everdivorced', 'Zposaffect', 'currsmokeYN', 'Zhopelessness', 'eversmokeYN', 'Zeduccat', 'modactivityYN', 'vigactivityYN', 'sumadultAE']
model_params['var_dict']['dataset']='Datasource'

# sl initialisation
start = float(sys.argv[1])
end = float(sys.argv[2])
seed_ind = int(sys.argv[3])
step=0.05
model='sl'
seeds = [1981236101, 56317909029, 91833472504,6588504046385,6747454181206,2069405004429]
var_set = model_params['domain_dict'][domain_name]
Eva_fixed_test_size_sl = pd.DataFrame(columns=['model', 'var_num','train_subset_size', 'test_auc_score', 'test_f1_score', 'test_pr_auc', 'test_pr_no_skill', 'test_efron_r2', 'test_ffc_r2','test_briern_r2', 'test_imv_r2'])

# main
for train_subset_size in np.arange(start, end, step):

    for sub_set_size in range(1, len(var_set) + 1):
        print(train_subset_size, sub_set_size)
        random.seed(seed_ind)  #
        model_params['domain_dict']['sub_set'] = random.sample(var_set, sub_set_size)
        seed = random.randint(a=1, b=1000000000000)
        model_params['seed'] = seed
        print(model_params['test_size'])
        superlearner = SuperLearner.superlearner(data=df,
                                                 train_subset_size=train_subset_size,
                                                 test_size=model_params['test_size'],
                                                 domain_list=model_params['domain_dict']['sub_set'],
                                                 y_colname=model_params['y_colname'],
                                                 k=model_params['k'],
                                                 random_state=model_params['random_state'])
        sl_eva = sl_only_eva(superlearner)
        # df_eva.loc[len(df_eva)] = [model, pr_auc, roc_auc, f1, efron, ffc, ip, imv_]
        Eva_fixed_test_size_sl.loc[len(Eva_fixed_test_size_sl)] = ['sl',
                                                                   sub_set_size,
                                                                   int(len(df) * train_subset_size),
                                                                   sl_eva['test_auc_score'],
                                                                   sl_eva['test_f1_score'],
                                                                   sl_eva['test_pr_auc'],
                                                                   sl_eva['test_pr_no_skill'],
                                                                   sl_eva['test_efron_r2'],
                                                                   sl_eva['test_ffc_r2'],
                                                                   sl_eva['test_briern_r2'],
                                                                   sl_eva['test_imv_r2']]

        Eva_fixed_test_size_sl.to_csv(f"/gpfs3/users/mills/qlr082/OX_thesis/results/asymptotics_sl_with_random_sampling_on_vars_seed_specified_{start}_{end}_{seed_ind}.csv",index=False)

# part 2 analysis

import pandas as pd
import os


files = [x for x in os.listdir(os.getcwd()+'/results') if x.startswith('asymptotics_sl_with_random_sampling_on_vars_seed_specified_')]
files.remove("asymptotics_sl_with_random_sampling_on_vars_seed_specified_0.csv")
df = pd.DataFrame()
for file in files:
    temp = pd.read_csv(os.getcwd()+f'/results/{file}')
    if file  == 'asymptotics_sl_with_random_sampling_on_vars_seed_specified_0.3_0.4_0.csv':
        temp = temp.loc[~temp['train_subset_size']== 22766]
    df = pd.concat([df,temp],axis=0)
    del(temp)

# plot
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

columns = []
for column in df.columns:
    if 'test' in column:
        columns.append(column)

df_to_plot = df.drop_duplicates(subset=['train_subset_size','var_num']).copy()
df_to_plot.reset_index(inplace=True,drop=True)


columns.remove('test_f1_score')
columns.remove('test_pr_no_skill')
columns.remove('test_auc_score')
columns.remove('test_briern_r2')

ploted_col_dict = {
                   'test_efron_r2': 'Efron R2',
                   'test_pr_auc': 'PR-AUC Score',
                   'test_ffc_r2': 'FFC R2',
                   'test_imv_r2': 'IMV'}


fig, axis = plt.subplots(2, 2)
fontsize_ticks = 13
fontsize_labels = 15
fig.subplots_adjust(left=0.08, bottom=0.08, top=0.95, right=0.99)
plt.rcParams['figure.figsize'] = [14, 10]

x_ticks = [22766, 28458, 34150, 39841, 45533, 51225, 56917]
count = 0
columns = ['test_pr_auc','test_imv_r2','test_efron_r2','test_ffc_r2']
for (m, n), subplot in np.ndenumerate(axis):
    metric = columns[count]
    pivot_table = df_to_plot.pivot('var_num', 'train_subset_size', metric)
    sns.heatmap(pivot_table, ax=axis[m, n], cmap=sns.dark_palette("#69d", reverse=False, as_cmap=True))

    axis[m, n].set_ylabel('Variable Number',fontsize = fontsize_labels)
    axis[m, n].set_xlabel('Training Set Size',fontsize = fontsize_labels)
    axis[m, n].set_title(ploted_col_dict[metric],size=15,weight="bold")  # , size=fontsize_labels)

    # set ticks
    axis[m, n].set_yticks([1, 5, 10, 15, 20, 26])
    axis[m, n].set_yticklabels([1, 5, 10, 15, 20, 26])
    axis[m, n].tick_params(axis='x', rotation=45)

    every_nth = 2
    for nth, label in enumerate(axis[m, n].xaxis.get_ticklabels()):
        if nth % every_nth != 0:
            label.set_visible(False)

    # axis[m, n].xaxis.set_major_locator(plt.MaxNLocator(8))

    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m, n].grid(axis='both', alpha=0.4)
    axis[m, n].set_axisbelow(True)



    count += 1
fig.tight_layout()


plt.savefig(pathlib.Path.cwd()/'graphs/asymptotics_sl.pdf')


