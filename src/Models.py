import pandas as pd

from src import Evaluate
import numpy as np
import xgboost as XGB
import sys
if sys.version=='3.7.16 (default, Jan 17 2023, 09:28:58) \n[Clang 14.0.6 ]':
    lightgbm=''
    CoxPHFitter=''
else:
    import lightgbm as LGB
    from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import random
random.seed(87785)
np.random.seed(87785)
print(f"LightGBM version: {LGB.__version__}")
print(f"XGBoost version: {XGB.__version__}")
print(f"NumPy version: {np.__version__}")


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['XGB_PRECISION'] = '1e-7'


def polynominal(df, order):
    for i in np.arange(2, order+1, 1):
        # print('order {}'.format(i))
        for column in df.columns:
            colname = column + '_' + str(i)
            temp=pd.DataFrame()
            temp[colname] = np.power(df[column], i)
            df = pd.concat([df, temp], axis=1)
    return df



class Model_fixed_test_size():
    def __init__(self, data, model_params, domain, model, train_subset_size, order):
        super(Model_fixed_test_size, self).__init__()

        # train test split

        test_size = model_params['test_size']
        if domain in list(model_params['domain_dict'].keys()):
            domain_list = model_params['domain_dict'][domain]
        else:
            domain_list = domain
        y_colname = model_params['y_colname']
        random_state = model_params['random_state']
        #print(f'seed is {random_state}')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[domain_list],
                                                                                data[y_colname],
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        # if we only need a subset of the training set train_subset_size!=0

        if "train_on_ELSA_first" in model_params.keys():
            print('train on ELSA first')
            n = int(train_subset_size * len(self.X_train))
            train_set_ELSA_proportion = len(self.X_train.loc[data['dataset'] == 2])
            if n <= train_set_ELSA_proportion:
                self.X_train = self.X_train.loc[data['dataset'] == 2].sample(n=n, random_state=random_state) # 2 is ELSA, 0 is HRS
            else:
                self.X_train = pd.concat([self.X_train.loc[data['dataset'] == 2],self.X_train.loc[data['dataset'] == 0 ].sample(n=n-train_set_ELSA_proportion, random_state=random_state)],axis=0)
        else:

            self.X_train = self.X_train.sample(n=int(train_subset_size * len(self.X_train)), random_state=random_state)

        self.y_train = self.y_train.loc[self.X_train.index]

        if 'sampWeight' in list(self.X_train.columns):
            self.samp_weight_control = True
            self.train_sample_weight = self.X_train['sampWeight']
            self.test_sample_weight = self.X_test['sampWeight']

        else:
            self.samp_weight_control = False


        if order != 0:
            # print('Before: the colnum of df is {}'.format(len(self.X_train.columns)))
            self.X_train = polynominal(self.X_train, order)
            self.X_test = polynominal(self.X_test, order)
            # print('After: the colnum of df is {}'.format(len(self.X_train.columns)))

        #  print('Train set number {}'.format(self.X_train.shape))
        if model == 'lgb':
            if self.samp_weight_control:
                self.model = LGB.LGBMClassifier(random_state=random_state, n_jobs=1,subsample=1.0,colsample_bytree=1.0)
                self.model.fit(X=self.X_train,
                               y=self.y_train,
                               sample_weight=self.train_sample_weight)
                print('test')
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:, 1]

                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]

            else:
                #print('no SampWeight')
                self.model = LGB.LGBMClassifier(random_state=random_state, n_jobs=1,subsample=1.0,colsample_bytree=1.0)
                self.model.fit(X=self.X_train,
                               y=self.y_train)
                #print('test')
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:,1]

                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:,1]

        if model == 'xgb':
            if self.samp_weight_control:
                # print('with sample weight')
                self.model = XGB.XGBClassifier(random_state=random_state,
                                               subsample=1.0,  # Use all rows for each boosting iteration
                                               colsample_bytree=1.0,  # Use all features for each tree
                                               colsample_bylevel=1.0,  # Use all features at each tree level
                                               colsample_bynode=1.0,  # Use all features for each tree node
                                               max_bin=256,  # Fix number of bins for feature discretization
                                               n_jobs=1,  # Single-threaded execution
                                               tree_method='exact',  # Exact tree construction
                                               grow_policy='depthwise',  # Fixed growth policy
                                               max_depth=6,  # Maximum depth of each tree
                                               learning_rate=0.1)#,use_label_encoder=False
                self.model.fit(X=self.X_train,
                               y=self.y_train,
                               sample_weight=self.train_sample_weight)
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:, 1]
                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]
            else:
                # print('without sample weight')
                self.model = XGB.XGBClassifier(random_state=random_state,
                                               subsample=1.0,  # Use all rows for each boosting iteration
                                               colsample_bytree=1.0,  # Use all features for each tree
                                               colsample_bylevel=1.0,  # Use all features at each tree level
                                               colsample_bynode=1.0,  # Use all features for each tree node
                                               max_bin=256,  # Fix number of bins for feature discretization
                                               n_jobs=1,  # Single-threaded execution
                                               tree_method='exact',  # Exact tree construction
                                               grow_policy='depthwise',  # Fixed growth policy
                                               max_depth=6,  # Maximum depth of each tree
                                               learning_rate=0.1)#,use_label_encoder=False#,use_label_encoder=False)
                self.model.fit(X=self.X_train,
                               y=self.y_train)
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:, 1]
                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]

        if model == 'logreg':
            self.model = LogisticRegression(random_state=random_state)
            self.model.fit(X=self.X_train,
                           y=self.y_train,
                           sample_weight=self.train_sample_weight)
            self.train_set_predict = self.model.predict(self.X_train)
            self.train_set_predict_prob = self.model.predict(self.X_train)[:, 1]
            self.test_set_predict = self.model.predict(self.X_test)
            self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]

        if model == 'cox':
            domain = list(
                set(domain_list + ['age', 'death', 'sampWeight', 'hhid', 'maleYN', 'blackYN', 'hispanicYN', 'otherYN',
                               'migrantYN']))
            self.X_train, self.X_test = train_test_split(data, test_size=test_size, random_state=87785)
            self.X_train = self.X_train[domain]
            self.X_test = self.X_test[domain]

            self.model = CoxPHFitter()
            self.model.fit(self.X_train, duration_col='age', event_col='death', weights_col='sampWeight',
                           cluster_col='hhid', robust=True)



# Logistic regression store coefficcient
def store_coef(count, k, model_name, seed, model, domains, df_coef,cv_name):

    NOK = str(count) + '/' + str(k)
    coef_dict = {'model': model_name, 'seed': seed,'fold_method': cv_name, 'k': k, 'NOK': NOK}
    for coef, feature in zip(model.coef_.T, domains):
        coef_dict[feature] = coef[0]
    temp = pd.DataFrame(coef_dict, index=[0])
    df_coef.loc[len(df_coef), ] = temp.loc[0, ]
    return df_coef

