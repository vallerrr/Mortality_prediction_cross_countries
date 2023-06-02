import os
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
start = sys.argv[1]
end = sys.argv[2]
step=0.5
model='sl'
var_set = model_params['domain_dict'][domain_name]
Eva_fixed_test_size_sl = pd.DataFrame(columns=['model', 'var_num','train_subset_size', 'test_auc_score', 'test_f1_score',
                 'test_pr_auc', 'test_pr_no_skill', 'test_efron_r2', 'test_ffc_r2','test_briern_r2', 'test_imv_r2'])

# main
for train_subset_size in np.arange(start, end, step):

    for sub_set_size in range(1, len(var_set) + 1):
        print(train_subset_size, sub_set_size)
        random.seed(1981236101)
        model_params['domain_dict']['sub_set'] = random.sample(var_set, sub_set_size)
        seed = random.randint(a=1, b=1000000000000)
        model_params['seed'] = seed
        print(model_params['test_size'])
        superlearner = SuperLearner.superlearner(data=df,
                                                 test_size=model_params['test_size'],
                                                 domain_list=model_params['domain_dict']['sub_set'],
                                                 y_colname=model_params['y_colname'],
                                                 k=model_params['k'],
                                                 random_state=model_params['random_state'])
        sl_eva = sl_only_eva(superlearner)
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

        Eva_fixed_test_size_sl.to_csv(f"/gpfs3/users/mills/qlr082/OX_thesis/results/asymptotics_sl_with_random_sampling_on_vars_seed_specified_{start}_{end}.csv",index=False)
