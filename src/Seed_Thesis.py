from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random

import matplotlib.pyplot as plt
import lightgbm as LGB
from src import Evaluate
from src import DataImport
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from pathlib import Path

df = DataImport.data_reader_by_us(bio=True)
test_size=0.3
seed = 1
domain_dict = DataImport.domain_dict()
domains = list(set(domain_dict['all_bio_adjusted']))

df_eval=pd.DataFrame(columns=['model','seed','roc_auc','pr_auc','f1','efron_r2','ffc_r2','imv','brier'])

for seed in np.arange(0,1000,1):
    X, test_X, y, test_y = train_test_split(df.drop('death', axis=1), df['death'], test_size=test_size,
                                            random_state=seed)
    model=LGB.LGBMClassifier()
    model.fit(X=X[domains],y=y)

    pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]

    Eval = Evaluate.evaluate_metric(test_y, pred, pred_prob, y)
    df_eval.loc[len(df_eval),] = [model, seed, Eval['roc_auc'],Eval['auc'], Eval['f1'],
                                  Eval['efron_r2'], Eval['ffc_r2'],Eval['imv'],Eval['brier']]


# df_eval.to_csv(Path.cwd()/'Seed/1fold1000seed.csv')

df_eval=pd.read_csv(Path.cwd()/'Seed/1fold1000seed.csv',index_col=0)


'''colums=['roc_auc', 'pr_auc', 'f1', 'efron_r2', 'ffc_r2', 'imv']
for column in colums:
    print(column,'&',round(df_eval[column].min(),3),'&',round(df_eval[column].max(),3),'\\\\')
'''


import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

fig,ax = plt.subplots(3,2)
plt.rcParams["figure.figsize"]=[12,10]
count=0
colums=['roc_auc', 'pr_auc', 'f1', 'efron_r2', 'ffc_r2', 'imv']
column_dict={'roc_auc':'ROC-AUC Score', 'pr_auc':'PR-AUC Score', 'f1':'F1', 'efron_r2':'Efron R2', 'ffc_r2':'FFC R2', 'imv':'IMV'}
fig.subplots_adjust(left=0.09, top=0.98, bottom=0.06, right=0.95)

colors = ['#001c54', '#E89818']
letter_fontsize = 24
label_fontsize = 16
for (m, n), subplot in np.ndenumerate(ax):

    sns.distplot(df_eval[colums[count]],
                 hist_kws={'facecolor': colors[0],'edgecolor': 'k','alpha': 0.6,},
                 kde_kws={'color': colors[1]}, ax=ax[m,n], bins=20)
    # ax[m,n].hist(df_eval[colums[count]],color=color_blue,alpha=0.75,bins=30,edgecolor='black')
    ax[m, n].set_xlabel(column_dict[colums[count]],fontsize=label_fontsize+1)
    ax[m, n].set_ylabel('Density',fontsize=label_fontsize + 1)
    ax[m,n].tick_params(axis='both', which='major', labelsize=label_fontsize)
    count+=1
    ax[m, n].spines['top'].set_visible(False)
    ax[m, n].spines['right'].set_visible(False)
plt.show()


plt.savefig(Path.cwd()/'graphs/seed.pdf')

for column in colums:
    print(column)
    print(df_eval[column].describe())
