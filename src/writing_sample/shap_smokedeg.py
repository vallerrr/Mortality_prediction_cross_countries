import os
import numpy as np
import pandas as pd
from pathlib import Path
from src import DataImport
import matplotlib.pyplot as plt
from src import Models
import shap
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src import Evaluate

test_size = 0.3
df = DataImport.data_reader_by_us(bio=False)
domains = DataImport.domain_dict()
var_dict = DataImport.variable_dict()



# variable preprocess

# smoke degree
temp = pd.DataFrame(columns=["eversmokeYN", "currsmokeYN", "smokedeg"])

temp['eversmokeYN']=df['eversmokeYN']
temp["currsmokeYN"] = df['currsmokeYN']
temp['smokedeg'] = -1



for i in temp.index:
    if (temp.loc[i, "eversmokeYN"] >=0 ) & (temp.loc[i, "currsmokeYN"] <= 0):  # past:1, now: -1, -> 1 most healthy
        temp.loc[i, "smokedeg"] = 1 # past smoker
    elif (temp.loc[i, "eversmokeYN"] <=0) & (temp.loc[i, "currsmokeYN"] <= 0):  # past:-1, now: -1,-> 2 mid healthy
        temp.loc[i, "smokedeg"] = 0  # non-smoker
    elif (temp.loc[i, "eversmokeYN"] <=0) & (temp.loc[i, "currsmokeYN"] >=0):  # past:-1, now: 1,-> 3 least healthy
        temp.loc[i, "smokedeg"] = 2
    else: # past:1, now: 1,-> 3 least healthy? ->no such
        temp.loc[i, "smokedeg"] = 3  # consistent smoker

temp['smokedeg'].value_counts()


df['eversmokeYN'] = temp["eversmokeYN"]
df['currsmokeYN'] = temp["currsmokeYN"]
df['smokedeg'] = temp['smokedeg']
var_dict['smokedeg'] = 'Smoker Degree'

domain = domains['all']+['smokedeg']
domain.remove('eversmokeYN')
domain.remove('currsmokeYN')


model = Models.Model_fixed_test_size(data=df, test_size=test_size, domain_list=domain, model='xgb',
                                     train_subset_size=1, order=0, y_colname='death')
evaluate=Evaluate.metric(model)
xgb_test = xgboost.DMatrix(model.X_test, label=model.y_test)


print('f1 is {}, pr_auc is {}, pr_no_skill is {}, roc_auc is {}'.format(evaluate.pr_f1, evaluate.pr_auc, evaluate.pr_no_skill, evaluate.auc_score))

# -------------------------------
# cross validation
# -------------------------------
params = {"objective": "binary:logistic", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10}
xgb_cv = xgboost.cv(dtrain=xgb_test, nfold=5, params=params, num_boost_round=50, early_stopping_rounds=10,
                    metrics="auc", as_pandas=True, seed=2022)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(model.model, model.X_test, model.y_test, cv=kfold)

print(results.mean(), results.std())



# -------------------------------
# | shap values                  |
# -------------------------------

# test set
explainer = shap.TreeExplainer(model.model)
shap_values_test = explainer(model.X_test)

# store the mean absolute shap value for each variable in a dictionary
shap_dic = {}
i = 0
while i < shap_values_test.values.shape[1]:
    sum_shap = 0
    for m in shap_values_test.values[:, i]:
        sum_shap += np.abs(m)
    shap_dic[shap_values_test.feature_names[i]] = sum_shap / 4084
    i += 1


# shap.dependence_plot("smokedeg",shap_values_test, model.X_test)


# beeswarm plot
fontsize_ticks = 20
fontsize_labels = 21
shap.summary_plot(shap_values_test, model.X_test, show=False, max_display=10, bar_fontzie=20, cmap='coolwarm')
fig = plt.gcf()
fig.set_figheight(10)
fig.set_figwidth(16)
fig.subplots_adjust(left=0.28, top=0.95, right=1.01, bottom=0.1)

ax = plt.gca()
plt.rc('legend', fontsize=25)
ax.set_xlabel('SHAP Value', fontsize=fontsize_labels)
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.set_yticklabels(ylabels)
plt.show()
#plt.savefig('beeswarm_smoke_degree.pdf')


# box plot of smokedeg
var = 'smokedeg'
ind = shap_values_test.feature_names.index(var)
shap_value = shap_values_test.values[:, ind]
value = list(shap_values_test.data[:, ind])

value_ind_1 = [i for i in np.arange(0, len(value),1) if value[i]==0.0]
value_ind_2 = [i for i in np.arange(0, len(value),1) if value[i]==1.0]
value_ind_3 = [i for i in np.arange(0, len(value),1) if value[i]==3.0]
shap_value_1 = [shap_value[i] for i in value_ind_1]
shap_value_2 = [shap_value[i] for i in value_ind_2]
shap_value_3 = [shap_value[i] for i in value_ind_3]


fontsize_ticks = 20
fontsize_labels = 22
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.09, top=0.95, bottom=0.18, right=0.95)
fig.set_figheight(3)
fig.set_figwidth(8)
colors = ['#3D78B0','#F28D44','#C73A32'] # '#68A857'
box_2= ax.boxplot([shap_value_1],positions=[1],labels=['Non-Smoker'],
                patch_artist=True,boxprops=dict(color=colors[1]), medianprops=dict(color=colors[1]), whiskerprops=dict(color=colors[1]), capprops=dict(color=colors[1]), showfliers=True)
box_1= ax.boxplot([shap_value_2],positions=[2],labels=['Past Smoker'],
                patch_artist=True,boxprops=dict(color=colors[0]), medianprops=dict(color=colors[0]), whiskerprops=dict(color=colors[0]), capprops=dict(color=colors[0]), showfliers=True)
box_3= ax.boxplot([shap_value_3],positions=[3],labels=['Consistent Smoker'],
                patch_artist=True,boxprops=dict(color=colors[2]), medianprops=dict(color=colors[2]), whiskerprops=dict(color=colors[2]), capprops=dict(color=colors[2]), showfliers=True)

box=[box_1,box_2,box_3]
# set the face color of boxes, which is the color of box
for i in [0, 1, 2]:
    for patch in box[i]['boxes']:
        patch.set(facecolor=colors[i], alpha=0.4)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Smoker Degree')
ax.set_ylabel('SHAP Value')
plt.show()
# plt.savefig(Path.cwd()/'graphs/smoke_box.pdf')

# version 2
# another attempt of box plot
df_smoke_shap=pd.DataFrame(columns=['shap','type'])
df_smoke_shap['shap'] = shap_value
for index,row in df.iterrows():
    if index in value_ind_1:
        df_smoke_shap.loc[index,'type']='Non-Smoker'
    elif index in value_ind_2:
        df_smoke_shap.loc[index, 'type'] = 'Past Smoker'
    else:
        df_smoke_shap.loc[index, 'type'] = 'Consistent Smoker'


import seaborn as sns

fontsize_ticks = 20
fontsize_labels = 22
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.09, top=0.95, bottom=0.18, right=0.95)
fig.set_figheight(3)
fig.set_figwidth(8)
sns.boxplot(x="type", y="shap",
                   data=df_smoke_shap, palette="Set2", dodge=True,ax=ax,color= ['#d53e4f', '#fdae61', '#3288bd'])
#sns.swarmplot(x="type", y="shap",data=df_smoke_shap, palette="Set2", dodge=True,ax=ax,color=".25")

ax.axhline(y=0, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Smoker Degree')
ax.set_ylabel('SHAP Value')
plt.show()
# end here

'''df['smokedeg'].value_counts()
# Interactions
shap_interaction_values = shap.TreeExplainer(model.model).shap_interaction_values(model.X_test)
shap.dependence_plot(
    ("age", "smokedeg"),
    shap_interaction_values, model.X_test
)
'''
