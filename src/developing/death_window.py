import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import Models
from src import Evaluate
from src import DataImport
from matplotlib.ticker import MaxNLocator
from pathlib import Path


domains = DataImport.domain_dict()
df_by_us = DataImport.data_reader_by_us(bio=True)

year_of_interest=2008
df = pd.DataFrame(df_by_us.loc[df_by_us['interview_year']==year_of_interest])
# df=df_by_us.csv.csv.copy()
df_deaths = df.loc[:,['death','death_year','deathYR','death_month']]
window_size=1
df_deaths['deathYR'].max()

scores = pd.DataFrame(columns=['time', 'type', 'f1', 'pr_no_skill','pr_auc', 'roc_auc', 'imv', 'brier','efron_r2','ffc_r2'])

window = list(np.arange(2008, 2020+window_size, window_size))
# create the y at each time and predict it
for time in window[1:]:
    print(time)
    index = window.index(time)
    y_colname_rolling = 'death_between'+str(window[index-1])+'-'+str(time)
    y_colname_recursive = 'death_before' + str(time)
    df[y_colname_rolling] = [1 if (x > window[index-1]) & (x <= time) else 0 for x in df['deathYR']]
    df[y_colname_recursive] = [1 if x <= time else 0 for x in df['deathYR']]

    model_rolling = Models.Model_fixed_test_size(data=df, test_size=0.3, domain_list=domains['all_bio_adjusted'],model='lgb',train_subset_size=1,order=0, y_colname=y_colname_rolling)
    model_recursive = Models.Model_fixed_test_size(data=df, test_size=0.3, domain_list=domains['all_bio_adjusted'],model='lgb',train_subset_size=1,order=0, y_colname=y_colname_recursive)

    if (1 in list(model_rolling.y_test)) & (0 in list(model_rolling.y_test)):
        eva_rolling = Evaluate.metric(model=model_rolling)
        line_rolling = {'time': time, 'type': 'rolling', 'f1': eva_rolling.pr_f1,
                        'pr_no_skill': eva_rolling.pr_no_skill, 'pr_auc': eva_rolling.pr_auc,
                        'roc_auc': eva_rolling.auc_score, 'imv': eva_rolling.imv, 'brier': eva_rolling.brier,
                        'efron_r2': eva_rolling.efron_rsquare, 'ffc_r2': eva_rolling.ffc_r2}
        scores = scores.append(line_rolling, ignore_index=True)
    else:
        print('skip time = {}, rolling window'.format(time))


    if (1 in list(model_recursive.y_test)) & (0 in list(model_recursive.y_test)):
        eva_recursive = Evaluate.metric(model=model_recursive)

        line_before = {'time': time, 'type': 'recursive', 'f1': eva_recursive.pr_f1, 'pr_no_skill': eva_recursive.pr_no_skill,'pr_auc':eva_recursive.pr_auc, 'roc_auc': eva_recursive.auc_score, 'imv': eva_recursive.imv, 'brier': eva_recursive.brier,
                       'efron_r2':eva_recursive.efron_rsquare,'ffc_r2':eva_recursive.ffc_r2}
        scores=scores.append(line_before, ignore_index=True)
    else:
        print('skip time = {}, recursive window'.format(time))


# PLOTS

# rolling window

scores_rolling = scores.loc[(scores['type'] == 'rolling')&(scores['time'] <= 2019)]
scores_recursive = scores.loc[(scores['type'] == 'recursive')&(scores['time'] <= 2019),]

color_yellow='#F1A52C'
color_blue='#001C5B'


fig, axis = plt.subplots(2,5)
plt.rcParams['figure.figsize']=[16,12]
fig.subplots_adjust(left=0.03, bottom=0.05,top=0.95,right=0.99)

column_to_display_dict={'roc_auc':'ROC-AUC Score','pr_auc':'PR-AUC Score','imv':'IMV Score','efron_r2':'Efron R2','ffc_r2':'FFC R2'}
count=0
for (m, n), subplot in np.ndenumerate(axis):
    if m==0:
        scores_display=scores_rolling.copy()
        title='Rolling Window\n'+str(year_of_interest)+'\n'
    else:
        scores_display=scores_recursive.copy()
        title = 'Recursive Window\n'+str(year_of_interest)+'\n'
    if count>=4:
        count-=5
    column = list(column_to_display_dict.keys())[count]

    # deal with pr_auc
    if column == 'pr_auc':
        scores_display.plot('time', ['pr_no_skill', 'pr_auc'], ax=axis[m,n],color=[color_yellow,color_blue])
    else:
        axis[m,n].plot(scores_display['time'],scores_display[column],color=color_blue)
    axis[m,n].set_title(column_to_display_dict[column])
    axis[m, n].set_xlabel('time')
    if n==2:
        axis[m, n].set_xlabel('time')
    # deal with 'imv'
    if column =='imv':
        axis[m,n].tick_params(axis='y', which='major', labelsize=8)
        axis[m,n].set_title(title+column_to_display_dict[column])
        axis[m, n].set_xlabel('')
    # deal with 'efron_r2'
    if (column =='efron_r2')&(m==0):
        axis[m, n].tick_params(axis='y', which='major', labelsize=8)

    axis[m, n].xaxis.set_major_locator(MaxNLocator(integer=True))
    count+=1

# plt.savefig(Path.cwd()/'graphs/death_window.pdf')
plt.show()


