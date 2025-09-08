import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import Models
from src import Evaluate

from matplotlib.ticker import MaxNLocator
from pathlib import Path
import matplotlib.gridspec as gridspec

from matplotlib.gridspec import SubplotSpec

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str,fontsize):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold',fontsize=fontsize)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


domains = params.domain_dict()
df_by_us = DataImport.data_reader_by_us(bio=False)


df=df_by_us.copy()
window_size=1


scores = pd.DataFrame(columns=['model','time', 'type', 'f1', 'pr_no_skill','pr_auc', 'roc_auc', 'imv', 'brier','efron_r2','ffc_r2'])

window = list(np.arange(0, 13+window_size, window_size))
# create the y at each time and predict it
for time in window[1:]:
    print(time)
    index = window.index(time)
    y_colname_rolling = 'death_between'+str(window[index-1])+'-'+str(time)
    y_colname_recursive = 'death_before' + str(time)
    time_span_2006 = [2006+window[index-1], 2006+time]
    time_span_2008 = [2008 + window[index - 1], 2008 + time]

    df[y_colname_rolling + '_06'] = [1 if (x > time_span_2006[0]) & (x <= time_span_2006[1]) &(y==2006)  else 0 for x,y in zip(df['deathYR'],df['interview_year'])]
    df[y_colname_rolling + '_08'] = [1 if (x > time_span_2008[0]) & (x <= time_span_2008[1])  &(y==2008)  else 0 for x,y in zip(df['deathYR'],df['interview_year'])]
    df[y_colname_rolling] = [1 if 1 in [x,y] else 0 for x,y in zip(df[y_colname_rolling + '_06'],df[y_colname_rolling + '_08'])]

    df[y_colname_recursive + '_06'] = [1 if (x<=time_span_2006[1]) &(y==2006) else 0 for x,y in zip(df['deathYR'],df['interview_year'])]
    df[y_colname_recursive + '_08'] = [1 if (x<=time_span_2008[1]) &(y==2008) else 0 for x,y in zip(df['deathYR'],df['interview_year'])]
    df[y_colname_recursive] = [1 if 1 in [x, y] else 0 for x, y in zip(df[y_colname_recursive + '_06'], df[y_colname_recursive + '_08'])]


    model_rolling = Models.Model_fixed_test_size(data=df, test_size=0.3, domain_list=domains['all'],model='xgb',train_subset_size=1,order=0, y_colname=y_colname_rolling)
    model_recursive = Models.Model_fixed_test_size(data=df, test_size=0.3, domain_list=domains['all'],model='xgb',train_subset_size=1,order=0, y_colname=y_colname_recursive)

    if (1 in list(model_rolling.y_test)) & (0 in list(model_rolling.y_test)):
        eva_rolling = Evaluate.metric(model=model_rolling)
        line_rolling = {'model':model_rolling,'time': time, 'type': 'rolling', 'f1': eva_rolling.pr_f1,
                        'pr_no_skill': eva_rolling.pr_no_skill, 'pr_auc': eva_rolling.pr_auc,
                        'roc_auc': eva_rolling.auc_score, 'imv': eva_rolling.imv, 'brier': eva_rolling.brier,
                        'efron_r2': eva_rolling.efron_rsquare, 'ffc_r2': eva_rolling.ffc_r2}
        scores = scores.append(line_rolling, ignore_index=True)
    else:
        print('skip time = {}, rolling window'.format(time))


    if (1 in list(model_recursive.y_test)) & (0 in list(model_recursive.y_test)):
        eva_recursive = Evaluate.metric(model=model_recursive)

        line_before = {'model':model_recursive,'time': time, 'type': 'recursive', 'f1': eva_recursive.pr_f1, 'pr_no_skill': eva_recursive.pr_no_skill,'pr_auc':eva_recursive.pr_auc, 'roc_auc': eva_recursive.auc_score, 'imv': eva_recursive.imv, 'brier': eva_recursive.brier,
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
fontsize_ticks = 17
fontsize_labels = 19

rows = 2
cols = 4

fig, axis = plt.subplots(rows, cols, figsize=(14,10))
fig.subplots_adjust(left=0.05, bottom=0.1,top=0.9,right=0.99,wspace=0.3,hspace=0.4)

column_to_display_dict={'pr_auc':'PR-AUC Score','imv':'IMV Score','efron_r2':'Efron R2','ffc_r2':'FFC R2'}
count=0
for (m, n), subplot in np.ndenumerate(axis):
    if m==0:
        scores_display=scores_rolling.copy()
    else:
        scores_display=scores_recursive.copy()
    if count>=3:
        count-=4
    column = list(column_to_display_dict.keys())[count]

    # deal with pr_auc
    if column == 'pr_auc':
        scores_display.plot('time', ['pr_no_skill', 'pr_auc'], ax=axis[m,n],color=[color_yellow,color_blue])
        axis[m,n].legend(labels=["IP",'PR'],fontsize=fontsize_labels-3)
    else:
        axis[m,n].plot(scores_display['time'],scores_display[column],color=color_blue)
    axis[m,n].set_title(column_to_display_dict[column],fontsize=fontsize_labels)
    axis[m, n].set_xlabel('years',fontsize=fontsize_labels)
    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks - 3)

    if n==2:
        axis[m, n].set_xlabel('years',fontsize=fontsize_labels)
    # deal with 'imv'
    if column =='imv':
        axis[m,n].tick_params(axis='y', which='major', labelsize=fontsize_ticks-3)
        axis[m,n].set_title(column_to_display_dict[column],fontsize=fontsize_labels)
        # axis[m, n].set_xlabel('')
    # deal with 'efron_r2'
    if (column =='efron_r2')&(m==0):
        axis[m, n].tick_params(axis='y', which='major', labelsize=fontsize_ticks-3)

    axis[m, n].xaxis.set_major_locator(MaxNLocator(integer=True))
    count+=1


grid = plt.GridSpec(rows, cols)
create_subtitle(fig, grid[0, ::], 'Rolling Window',fontsize=fontsize_labels)
create_subtitle(fig, grid[1, ::], 'Recursive Window',fontsize=fontsize_labels)

textstr="Note: The PR and IP stand for PR-AUC score and Insample Prevalence."

plt.figtext(0.02, 0.01, textstr, fontsize=fontsize_labels-2)
# plt.tight_layout()
plt.savefig(Path.cwd()/'graphs/death_window.pdf')
plt.show()
#



'''
df_by_us.csv.csv=DataImport.data_reader_by_us(bio=False)
df_by_author=DataImport.data_reader(bio=False)


df_by_author['death'].value_counts()
df_by_author['deathYR'].max()


df_by_author.drop(columns=['sampWeight'],inplace=True)
df_by_us.csv.csv['death_before_2015']=[1 if (x==1)&(y<=2018) else 0 for x,y in zip(df_by_us.csv.csv['death'],df_by_us.csv.csv['deathYR'])]
df_by_us.csv.csv['death_before_2015'].value_counts()

model = Models.Model_fixed_test_size(data=df, test_size=0.3, domain_list=domains['bio'],model='xgb',train_subset_size=1,order=0, y_colname='death')
evaluate = Evaluate.metric(model=model)

model_2 = Models.Model_fixed_test_size(data=df_by_us.csv.csv, test_size=0.3, domain_list=domains['all'],model='xgb',train_subset_size=1,order=0, y_colname='death_before_2015')
evaluate_2 = Evaluate.metric(model=model_2)

df_by_us.csv.csv['age'].median()
df_by_us.csv.csv.loc[df_by_us.csv.csv['death']==1,'age'].median()



x=[column if 'between' in column else None  for column in df.columns]
window

columns = ['death_between{}-{}'.format(window[window.index(i)-1],window[i]) for i in window[1:]]

rolling_window_check = df.loc[:,columns]
death_sum=0
for col in columns:
    deaths=rolling_window_check[col].sum()
    death_sum+=deaths
    print(deaths,death_sum)

df['death'].value_counts()
df['death_before13'].value_counts()
'''
'''
#SHAP Section
import shap

shap_values=pd.DataFrame(columns=['time']+domains['all'])

row=0
score_shap=scores_recursive.copy()
while row < score_shap.shape[0]:
    model = score_shap.iloc[row,0]
    time= score_shap.iloc[row,1]
    explainer = shap.TreeExplainer(model.model)
    shap_values_test = explainer(model.X_test)

    shap_dic = {'time':time}
    i = 0

    while i < shap_values_test.values.shape[1]:
        sum_shap = 0
        for m in shap_values_test.values[:, i]:
            sum_shap += np.abs(m)
        shap_dic[shap_values_test.feature_names[i]] = sum_shap / shap_values_test.values.shape[0]
        i += 1

    shap_values=shap_values.append(shap_dic,ignore_index=True)
    row+=1


# store the mean absolute shap value for each variable in a dictionary

col_to_display_shap =['age','maleYN','vigactivityYN','eversmokeYN','currsmokeYN','ZwealthT','ZincomeT','Zanxiety','rocc','Zneighdisorder']
shap_values.plot('time',col_to_display_shap)
plt.show()
'''
