import os

import numpy as np
import pandas as pd

from src import DataImport
import matplotlib.pyplot as plt
from src import Models
import shap
import xgboost
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src import Evaluate

test_size = 0.3
df = DataImport.data_reader()
domains = DataImport.domain_dict()
var_dict = DataImport.variable_dict()

# variable preprocess

# smoke degree
temp = pd.DataFrame(columns=["eversmokeYN", "currsmokeYN", "smokedeg"])

temp["eversmokeYN"] = df['eversmokeYN'].apply(lambda x: 1.00 if x > 0 else -1)
temp["currsmokeYN"] = df['currsmokeYN'].apply(lambda x: 1.00 if x > 0 else -1)
temp['smokedeg'] = -1

for i in temp.index:
    if (temp.loc[i, "eversmokeYN"] == 1) & (temp.loc[i, "currsmokeYN"] == -1):  # past:1, now: -1, -> 1 most healthy
        temp.loc[i, "smokedeg"] = 1
    elif (temp.loc[i, "eversmokeYN"] == -1) & (temp.loc[i, "currsmokeYN"] == -1):  # past:-1, now: -1,-> 2 mid healthy
        temp.loc[i, "smokedeg"] = 2
    elif (temp.loc[i, "eversmokeYN"] == -1) & (temp.loc[i, "currsmokeYN"] == 1):  # past:-1, now: 1,-> 3 least healthy
        temp.loc[i, "smokedeg"] = 3

df['eversmokeYN'] = temp["eversmokeYN"]
df['currsmokeYN'] = temp["currsmokeYN"]
df['smokedeg'] = temp['smokedeg']
var_dict['smokedeg'] = 'Smoke Degree'

# normalise (mean=0,std =1)
# df['smokedeg'] = (df['smokedeg'] -df['smokedeg'].mean())/df['smokedeg'].std()
# df['smokedeg'].value_counts()

# df['everunemployed'] = df['everunemployed'].apply(lambda x: 1.00 if x > 0 else -1.00)

# other binary variables
# f1 is 0.47150823658402086, pr_auc is 0.5853536639623846, pr_no_skill is 0.20102840352595494, roc_auc is 0.8393748881402152
# 0.8227230182158543 0.005543001807696724
# binary_var_lst = ['relocate','finhelp','everfoodstamp','everrent','evermedicaid','everunemployed','everfoodinsec','vigactivityYN',
#                  'modactivityYN','alcoholYN','sleepYN','blackYN','otherYN','migrantYN']
'''
binary_var_lst = ['fatherunemp', 'relocate', 'finhelp']
for var in binary_var_lst:
   df[var] = df[var].apply([lambda x: 1.00 if x > 0 else -1])
'''

# for var in domains['all']:
#    print(df[var].value_counts())


domain = domains['all']+['smokedeg']
domain.remove('eversmokeYN')
domain.remove('currsmokeYN')


model = Models.Model_fixed_test_size(data=df, test_size=test_size, domain_list=domain, model='xgb',
                                     train_subset_size=1, order=0)

xgb_test = xgboost.DMatrix(model.X_test, label=model.y_test)

pred_prob, pred_label = model.test_set_predict_prob, model.test_set_predict
y_test, sample_weight = model.y_test, model.test_sample_weight

# pr part
precision, recall, _ = precision_recall_curve(y_test, pred_prob, sample_weight=sample_weight)
pr_f1, pr_auc = f1_score(y_test, pred_label, sample_weight=sample_weight), auc(recall, precision)
pr_no_skill = len(y_test[y_test == 1]) / len(y_test)
r_score = Evaluate.r2(model.y_test, model.test_set_predict)
brier = Evaluate.brier(model.y_test, model.test_set_predict_prob)

# roc
roc_no_skill = 0.5
auc_score = roc_auc_score(y_test, pred_prob, sample_weight=sample_weight)



print('f1 is {}, pr_auc is {}, pr_no_skill is {}, roc_auc is {}'.format(pr_f1, pr_auc, pr_no_skill, auc_score))

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

plt.savefig('beeswarm_smoke_degree.pdf')


# box plot of smokedeg
var = 'smokedeg'
ind = shap_values_test.feature_names.index(var)
shap_value = shap_values_test.values[:, ind]
value = list(shap_values_test.data[:, ind])

value_ind_1 = [i for i in np.arange(0,len(value),1) if value[i]==1.0]
value_ind_2 = [i for i in np.arange(0,len(value),1) if value[i]==2.0]
value_ind_3 = [i for i in np.arange(0,len(value),1) if value[i]==3.0]
shap_value_1 = [shap_value[i] for i in value_ind_1]
shap_value_2 = [shap_value[i] for i in value_ind_2]
shap_value_3 = [shap_value[i] for i in value_ind_3]

fontsize_ticks = 20
fontsize_labels = 22
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.09, top=0.95, bottom=0.08, right=0.95)
fig.set_figheight(6)
fig.set_figwidth(8)
colors = ['#3D78B0','#F28D44','#C73A32'] # '#68A857'
box_1= ax.boxplot([shap_value_1],positions=[1],labels=['Past Smoker'],
                patch_artist=True,boxprops=dict(color=colors[0]), medianprops=dict(color=colors[0]), whiskerprops=dict(color=colors[0]), capprops=dict(color=colors[0]), showfliers=False)
box_2= ax.boxplot([shap_value_2],positions=[2],labels=['Non-Smoker'],
                patch_artist=True,boxprops=dict(color=colors[1]), medianprops=dict(color=colors[1]), whiskerprops=dict(color=colors[1]), capprops=dict(color=colors[1]), showfliers=False)
box_3= ax.boxplot([shap_value_3],positions=[3],labels=['Recent Smoker'],
                patch_artist=True,boxprops=dict(color=colors[2]), medianprops=dict(color=colors[2]), whiskerprops=dict(color=colors[2]), capprops=dict(color=colors[2]), showfliers=False)

box=[box_1,box_2,box_3]
# set the face color of boxes, which is the color of box
for i in [0, 1, 2]:
    for patch in box[i]['boxes']:
        patch.set(facecolor=colors[i], alpha=0.4)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Smoke Degree')
ax.set_ylabel('SHAP Value')
# plt.show()
plt.savefig('smoke_box.pdf')

# end here
'''

# summary bar plot
max_display = 11
fontsize_ticks = 20
fontsize_labels = 21
sum_features = 'Sum of ' + str(63 - max_display) + ' other features'
shap.plots.bar(shap_values_test, show=False, max_display=max_display)
fig = plt.gcf()
# 26，19
fig.set_figheight(10)
fig.set_figwidth(16)
# fig.subplots_adjust(left=0.4, top=0.99, bottom=0.04,right=0.95)
fig.subplots_adjust(left=0.3, top=0.99, bottom=0.04, right=0.95)

ax = plt.gca()
var_dict[sum_features] = sum_features
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.set_ylabel('Input Factors', fontsize=fontsize_labels)
ax.set_xlabel('mean(|SHAP Value|)', fontsize=fontsize_labels)
plt.show()
# plt.savefig('mean_shap_all.pdf')
# end here








# SHAP for certain variables

variable_to_display = [ 'smokedeg','ZincomeT', 'ZwealthT', 'Zanxiety', 'Zperceivedconstraints', 'Zconscientiousness']
lim_dic = {"age": [50, 100], "ZincomeT": [-6, 1], "ZwealthT": [-3.5, 1], 'Zanxiety': [-1.5, 4.5],
           'Zperceivedconstraints': [-1.5, 3.5], 'Zconscientiousness': [-2.5, 5], 'smokedeg':[0,4]}
i = 0
fontsize_ticks = 20
fontsize_labels = 21
figure, axis = plt.subplots(3, 2)
figure.subplots_adjust(left=0.08, top=0.95, bottom=0.08, right=0.95)
figure.set_figheight(16)
figure.set_figwidth(18)

for (m, n), subplot in np.ndenumerate(axis):
    var = variable_to_display[i]
    ind = shap_values_test.feature_names.index(var)
    shap_value = shap_values_test.values[:, ind]
    value = shap_values_test.data[:, ind]
    axis[m, n].scatter(value, shap_value, c=value, s=[25] * len(value), cmap='coolwarm',
                       norm=plt.Normalize(vmin=lim_dic[var][0], vmax=lim_dic[var][1]))
    axis[m, n].set_xlim(lim_dic[var][0], lim_dic[var][1])
    axis[m, n].grid(axis='y', alpha=0.4, linestyle='dashed')
    axis[m, n].axhline(y=0, color='red', linestyle='--', alpha=0.6)
    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m, n].set_ylabel('SHAP value', fontsize=fontsize_labels)
    axis[m, n].set_xlabel(var_dict[var], fontsize=fontsize_labels)
    axis[m, n].set_axisbelow(True)

    i += 1

plt.show()
# plt.savefig('ws2_plot.pdf')
'''
