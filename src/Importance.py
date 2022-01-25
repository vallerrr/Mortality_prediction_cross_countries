import DataImport
import xgboost
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = DataImport.data_reader()
domains = DataImport.domain_dict()
var_dict = DataImport.variable_dict()

# variables process
df['ZincomeT']= -1* df['ZincomeT']
df['ZincomeT'].value_counts()
var_dict['ZincomeT'] = 'Income'

# create interactions
'''
new_var = ['drinkwealth','drinkincome','eversmokewealth','currsmokewealth']

df[new_var[0]] = df['alcoholYN'] * df['ZwealthT']
var_dict[new_var[0]] = 'Alcohol Abuse * Lower Wealth'

df[new_var[1]] = df['alcoholYN'] * df['ZincomeT']
var_dict[new_var[1]] = 'Alcohol Abuse * Income'


df[new_var[2]] = df['eversmokeYN'] * df['ZwealthT']
var_dict[new_var[2]] = 'Ever Smoke * Lower Wealth'

df[new_var[3]] = df['currsmokeYN'] * df['ZwealthT']
var_dict[new_var[3]] = 'Current Smoke * Lower Wealth'
'''

domains.keys()
domain_list = list(set(domains['adult_SES'] + domains['demographic'] + domains['behavioral']))


test_size = 0.3
X, test_x, y, test_y = train_test_split(df.drop('death', axis=1), df['death'], test_size=test_size)



# create the model

xgb_full = xgboost.DMatrix(df[domain_list], label=df['death'])
xgb_train = xgboost.DMatrix(X[domain_list], label=y)
xgb_test = xgboost.DMatrix(test_x[domain_list], label=test_y)


'''
params = {
    "eta": 0.002,
    "max_depth": 3,
    "objective": "survival:cox",
    "subsample": 0.5
}

#model_train = xgboost.train(params, xgb_train, 10000, evals=[(xgb_test, "test")], verbose_eval=1000)
#model = xgboost.train(params, xgb_full, 1000, evals=[(xgb_full, "test")], verbose_eval=1000)
'''

model = xgboost.XGBClassifier()
model.fit(X=X[domain_list], y=y, sample_weight=X['sampWeight'])


#make predictions
predicts = model.predict_proba(test_x[domain_list])



explainer = shap.TreeExplainer(model)
shap_values = explainer(xgb_train)


shap.summary_plot(shap_values, X[domain_list], show=False,max_display=70)
fig = plt.gcf()
fig.set_figheight(12)
fig.set_figwidth(28)
ax = plt.gca()
ax.set_ylabel('Inputparameter', fontsize=16)
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)
plt.show()


'''shap.plots.bar(shap_values, show=False)
fig = plt.gcf()
fig.set_figheight(12)
fig.set_figwidth(10)
ax = plt.gca()
ax.set_ylabel('Inputparameter', fontsize=16)
ylabels = [var_dict[y_tick.get_text()] for y_tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels)
plt.show()
'''
