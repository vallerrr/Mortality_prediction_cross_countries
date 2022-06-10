from src import Models
from src import DataImport
from src import Evaluate
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def asymptotics(model_set, start, end, step, domains):
    Eva_fixed_test_size = pd.DataFrame(
        columns=['model', 'train_subset_size', 'train_auc_score', 'test_auc_score', 'train_f1_score', 'test_f1_score','test_pr_auc','test_pr_no_skill',
                 'train_efron_r2', 'test_efron_r2',
                 'train_ffc_r2', 'test_ffc_r2',
                  'test_briern_r2',  'test_imv_r2'])

    for model in model_set:
        print('Model: {}'.format(model))
        for train_subset_size in np.arange(start, end, step):
            print('train set size = {}'.format(train_subset_size))
            model_ = Models.Model_fixed_test_size(df, 0.3, domains, model, train_subset_size, polynomial,y_colname='death')
            eva=Evaluate.metric(model_)
            Eva_fixed_test_size.loc[len(Eva_fixed_test_size)] = [model,
                                                                 len(model_.X_train),

                                                                 # AUC score
                                                                 eva.test_roc_auc_score_label,
                                                                 eva.auc_score,

                                                                 # F1 score
                                                                 eva.train_f1_score_label,
                                                                 eva.pr_f1,

                                                                 # PR-AUC score
                                                                 eva.pr_auc,
                                                                 eva.pr_no_skill,

                                                                 # Efron_r2
                                                                 Evaluate.efron_rsquare(model_.y_train,
                                                                                        model_.train_set_predict_prob),
                                                                 eva.efron_rsquare,
                                                                 # ffc_r2
                                                                 Evaluate.ffc_rsquare(model_.y_train,
                                                                                      model_.train_set_predict_prob,
                                                                                      model_.y_train),

                                                                 eva.ffc_r2,
                                                                 # brier
                                                                 eva.brier,
                                                                 eva.imv]

    return Eva_fixed_test_size

random.seed(52194)
pd.set_option('precision', 9)

# read data
polynomial = 0

df = DataImport.data_reader_by_us(bio=True)
domain_dict = DataImport.domain_dict()
domains = list(set(domain_dict['all']))

domain_dict.keys()
# TODO: Add Domain

model_set = ['lgb']  # 'xgb', 'logreg']

temp=np.arange(0.3,1.005,0.005)
Eva_fixed_test_size = asymptotics(model_set=model_set, start=0.3, end=1.005, step=0.005, domains=domains)
# Eva_fixed_test_size_new = asymptotics(model_set=model_set, start=0.9, end=1, step=0.005, domains=domains)
# auc score
# Eva_fixed_test_size.to_csv(Path.cwd()/'result_csv/asymptotics_lgb.csv')

# Eva_fixed_test_size=pd.read_csv(Path.cwd()/'result_csv/asymptotics_lgb.csv',index_col=0)
columns =[]
for column in Eva_fixed_test_size.columns:
    if 'test' in column:
        columns.append(column)


df_to_plot=Eva_fixed_test_size.copy()


columns.remove('test_f1_score')
columns.remove('test_pr_no_skill')

ploted_col_dict = {'test_auc_score': 'ROC-AUC Score',
 # 'test_f1_score':'F1 Score',

 'test_efron_r2':'Efron R2',
 'test_pr_auc':'PR-AUC Score',
 'test_ffc_r2':'FFC R2',
 'test_briern_r2':'Brier Score','test_imv_r2': 'IMV'}

color_yellow='#F1A52C'
color_blue='#001C5B'


import seaborn
fig, axis = plt.subplots(2,3)
fontsize_ticks = 14
fontsize_labels = 15
fig.subplots_adjust(left=0.08, bottom=0.08,top=0.95,right=0.99)
plt.rcParams['figure.figsize']=[14,10]

count=0
for (m, n), subplot in np.ndenumerate(axis):
    seaborn.regplot(x=df_to_plot['train_subset_size'].astype(float),y=df_to_plot[columns[count]].astype(float),color=color_blue,ax=axis[m, n])
    axis[m, n].set_ylabel(ploted_col_dict[columns[count]],size=fontsize_labels)
    axis[m,n].set_xlabel('Training Set Size',size=fontsize_labels)

    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m,n].grid(axis='both',alpha=0.4)
    axis[m, n].set_axisbelow(True)
    count+=1
plt.show()
# plt.savefig(Path.cwd()/'graphs/asymptotics.pdf')



