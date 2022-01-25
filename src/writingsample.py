import numpy as np
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
# income
'''
df['ZincomeT'].describe()
df['ZincomeT'] = -1*df['ZincomeT']
var_dict['ZincomeT'] = 'Income'

# wealth
df['ZwealthT'].describe()
df['ZwealthT']=-1*df['ZwealthT']
var_dict['ZwealthT'] = 'Wealth'
'''
# everunemployed
df['everunemployed'].describe()
df['everunemployed']=df['everunemployed'].apply(lambda x: 1.00 if x > 0 else -1.00)

# Zanxiety
df['Zanxiety'].describe()
#df['Zanxiety'].hist()
#plt.show()

# Zperceivedconstraints
df['Zperceivedconstraints'].describe()
#df['Zperceivedconstraints'].hist()

#Zconscientiousness
df['Zconscientiousness'].describe()
#df['Zconscientiousness'].hist()
#plt.show()


model = Models.Model_fixed_test_size(data=df, test_size=test_size, domain_list=domains['all'], model='xgb',
                                      train_subset_size=1, order=0)


xgb_test = xgboost.DMatrix(model.X_test, label=model.y_test)

pred_prob, pred_label = model.test_set_predict_prob, model.test_set_predict
y_test, sample_weight = model.y_test, model.test_sample_weight

# pr part
precision, recall, _ = precision_recall_curve(y_test, pred_prob, sample_weight=sample_weight)
pr_f1, pr_auc = f1_score(y_test, pred_label, sample_weight=sample_weight), auc(recall, precision)
pr_no_skill = len(y_test[y_test == 1]) / len(y_test)
r_score = Evaluate.r2(model.y_test,model.test_set_predict)
brier = Evaluate.brier(model.y_test,model.test_set_predict_prob)
# roc
roc_no_skill = 0.5
auc_score = roc_auc_score(y_test, pred_prob, sample_weight=sample_weight)


model.model

print('f1 is {}, pr_auc is {}, pr_no_skill is {}, roc_auc is {}'.format(pr_f1,pr_auc,pr_no_skill,auc_score))


# -------------------------------
# cross validation
# -------------------------------
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
xgb_cv = xgboost.cv(dtrain=xgb_test, nfold=5, params=params, num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=2022)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
results = cross_val_score(model.model, model.X_test, model.y_test, cv=kfold)

print(results.mean(),results.std())

# -------------------------------
#| shap values                  |
# -------------------------------

# test set
explainer = shap.TreeExplainer(model.model)
shap_values_test = explainer(model.X_test)


# summary scatter plot
fontsize_ticks = 20
fontsize_labels = 21
shap.summary_plot(shap_values_test, model.X_test, show=False, max_display=10, bar_fontzie=20,cmap='coolwarm')
fig = plt.gcf()
fig.set_figheight(10)
fig.set_figwidth(16)
fig.subplots_adjust(left=0.28, top=0.95,right=1.01,bottom=0.1)

ax = plt.gca()
plt.rc('legend', fontsize=25)
ax.set_xlabel('SHAP Value', fontsize=fontsize_labels)
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.set_yticklabels(ylabels)

#plt.show()
plt.savefig('summary_shap.pdf')
# end here



# summary bar plot
max_display = 11
fontsize_ticks = 20
fontsize_labels = 21
sum_features = 'Sum of '+str(63-max_display)+' other features'
shap.plots.bar(shap_values_test, show=False, max_display=max_display)
fig = plt.gcf()
# 26，19
fig.set_figheight(10)
fig.set_figwidth(16)
#fig.subplots_adjust(left=0.4, top=0.99, bottom=0.04,right=0.95)
fig.subplots_adjust(left=0.3, top=0.99, bottom=0.04,right=0.95)

ax = plt.gca()
var_dict[sum_features] = sum_features
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)
ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax.set_ylabel('Input Factors', fontsize=fontsize_labels)
ax.set_xlabel('mean(|SHAP Value|)', fontsize=fontsize_labels)
plt.show()
#plt.savefig('mean_shap_all.pdf')
# end here



# SHAP for certain variables

variable_to_display = ['age','ZincomeT','ZwealthT','Zanxiety','Zperceivedconstraints','Zconscientiousness']
lim_dic={"age":[50,100],"ZincomeT":[-6, 1],"ZwealthT":[-3.5, 1],'Zanxiety':[-1.5, 4.5],'Zperceivedconstraints':[-1.5, 3.5],'Zconscientiousness':[-2.5, 5]}
i = 0
fontsize_ticks = 20
fontsize_labels = 21
figure, axis = plt.subplots(3, 2)
figure.subplots_adjust(left=0.08, top=0.95, bottom=0.08,right=0.95)
figure.set_figheight(16)
figure.set_figwidth(18)

for (m, n), subplot in np.ndenumerate(axis):
    var = variable_to_display[i]
    ind = shap_values_test.feature_names.index(var)
    shap_value = shap_values_test.values[:, ind]
    value = shap_values_test.data[:, ind]
    axis[m, n].scatter(value, shap_value, c=value, s=[25] * len(value), cmap='coolwarm', norm = plt.Normalize(vmin=lim_dic[var][0], vmax=lim_dic[var][1]))
    axis[m, n].set_xlim(lim_dic[var][0], lim_dic[var][1])
    axis[m, n].grid(axis='y', alpha=0.4, linestyle='dashed')
    axis[m, n].axhline(y=0, color='red', linestyle='--', alpha=0.6)
    axis[m, n].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    axis[m, n].spines['top'].set_visible(False)
    axis[m, n].spines['right'].set_visible(False)
    axis[m, n].set_ylabel('SHAP value', fontsize=fontsize_labels)
    axis[m, n].set_xlabel(var_dict[var], fontsize=fontsize_labels)
    axis[m, n].set_axisbelow(True)

    i+=1

#plt.show()

plt.savefig('ws2_plot.pdf')









