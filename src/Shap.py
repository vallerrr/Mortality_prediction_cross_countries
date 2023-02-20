import numpy as np
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
# df = DataImport.data_reader(bio=False)
domains = DataImport.domain_dict()
var_dict = DataImport.variable_dict()



def model_performance(data, test_size, domain_list):
    model = Models.Model_fixed_test_size(data=data, test_size=test_size,
                                         domain_list=domain_list, model='xgb',
                                         train_subset_size=1, order=0, y_colname='death')

    evas = Evaluate.metric(model)
    pr_f1 = evas.pr_f1
    pr_auc = evas.pr_auc
    pr_no_skill = evas.pr_no_skill
    auc_score = evas.auc_score

    print('f1 is {}, pr_auc is {}, pr_no_skill is {}, roc_auc is {}'.format(pr_f1, pr_auc, pr_no_skill, auc_score))
    return model


model = model_performance(data=df, test_size=0.3, domain_list=domains['all'])
evaluate = Evaluate.metric(model=model)

'''# only keep rows in the df_bio
df_bio = DataImport.data_reader_by_us(bio=True)
temp_df= df.loc[df['hhidpn'].isin(df_bio['hhidpn'])]
temp_model = model_performance(data=temp_df, test_size=0.3,domain_list=domains['all'])
'''
# ---------------------------------------------------------------------------------------------
# cross validation
# ---------------------------------------------------------------------------------------------
params = {"objective": "binary:logistic", 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
xgb_test = xgboost.DMatrix(model.X_test, label=model.y_test)
xgb_cv = xgboost.cv(dtrain=xgb_test, nfold=5, params=params, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=2022)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(model.model, model.X_test, model.y_test, cv=kfold)

print(results.mean(), results.std())

# -------------------------------
# | shap values                  |
# -------------------------------

# test set
explainer = shap.TreeExplainer(model_2.model)
shap_values_test = explainer(model.X_test)

# store the mean absolute shap value for each variable in a dictionary
shap_dic = {}
i = 0
while i < shap_values_test.values.shape[1]:
    sum_shap = 0
    for m in shap_values_test.values[:, i]:
        sum_shap += np.abs(m)
    shap_dic[shap_values_test.feature_names[i]] = sum_shap / shap_values_test.values.shape[0]
    i += 1

var_dict['ZincomeT'] = 'Income'
var_dict['ZwealthT'] = 'Wealth'
# summary scatter plot
fontsize_ticks = 20
fontsize_labels = 21
shap.summary_plot(shap_values_test, model.X_test, show=False, max_display=10, cmap='coolwarm')
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

plt.savefig(Path.cwd() / 'graphs/summary_shap.pdf')
plt.show()
# end here


color_blue = '#001C5B'
# summary bar plot
max_display = 61
fontsize_ticks = 20
fontsize_labels = 21
sum_features = 'Sum of ' + str(62 - max_display) + ' other features'
shap.plots.bar(shap_values_test, show=False, max_display=max_display)
fig = plt.gcf()
# 26，19
fig.set_figheight(26)
fig.set_figwidth(19)
# fig.subplots_adjust(left=0.4, top=0.99, bottom=0.04,right=0.95)
fig.subplots_adjust(left=0.3, top=0.99, bottom=0.04, right=0.95)

ax = plt.gca()
var_dict[sum_features] = sum_features

ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
# xax.set_ylabel('Input Factors', fontsize=fontsize_labels)
ax.set_xlabel('mean(|SHAP Value|)', fontsize=fontsize_labels)
fig.tight_layout()
# plt.savefig(Path.cwd()/'graphs/mean_shap_top10.pdf')
plt.savefig(Path.cwd() / 'graphs/mean_shap_all.pdf')

plt.show()
# end here

# SHAP for certain variables

variable_to_display = ['age', 'ZincomeT', 'ZwealthT', 'Zanxiety', 'Zneuroticism', 'Zhopelessness', 'maleYN', 'vigactivityYN', 'eversmokeYN']

i = 0
fontsize_ticks = 20
fontsize_labels = 21
figure, axis = plt.subplots(5, 2)
figure.subplots_adjust(left=0.08, top=0.95, bottom=0.08, right=0.95)
figure.set_figheight(16)
figure.set_figwidth(18)

for (m, n), subplot in np.ndenumerate(axis):
    var = variable_to_display[i]
    ind = shap_values_test.feature_names.index(var)
    shap_value = shap_values_test.values[:, ind]
    value = shap_values_test.data[:, ind]

    lim_upper, lim_lower = int(value.max()) + 1, int(value.min()) - 1
    axis[m, n].scatter(value, shap_value, c=value, s=[25] * len(value), cmap='coolwarm', norm=plt.Normalize(vmin=lim_lower, vmax=lim_upper))
    axis[m, n].set_xlim(lim_lower, lim_upper)
    axis[m, n].grid(axis='y', alpha=0.4, linestyle='dashed')
    axis[m, n].axhline(y=0, color='red', linestyle='--', alpha=0.6)
    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m, n].set_ylabel('SHAP value', fontsize=fontsize_labels)
    axis[m, n].set_xlabel(var_dict[var], fontsize=fontsize_labels)
    axis[m, n].set_axisbelow(True)

    i += 1

# plt.show()
# plt.savefig(Path.cwd()/'graphs/contious_shap.pdf')
