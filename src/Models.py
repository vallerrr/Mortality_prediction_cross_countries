import pandas as pd
from src import DataImport
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


# domain 1 : demographic
## 1.1 forecast death

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
    def __init__(self, data, test_size, domain_list, model, train_subset_size, order,y_colname,random_state = 4):
        super(Model_fixed_test_size, self).__init__()
        # train test split
        print(f'seed is {random_state}')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(y_colname, axis=1),
                                                                                data[y_colname], test_size=test_size, random_state=random_state)
        self.X_train = self.X_train.sample(n=int(train_subset_size * len(self.X_train)), random_state=random_state)

        self.y_train = self.y_train.loc[self.X_train.index]
        if 'sampWeight' in list(self.X_train.columns):
            self.samp_weight_control = True
            self.train_sample_weight = self.X_train['sampWeight']
            self.test_sample_weight = self.X_test['sampWeight']

        else:
            self.samp_weight_control = False
        self.X_train = self.X_train[domain_list]
        self.X_test = self.X_test[domain_list]

        if order != 0:
            # print('Before: the colnum of df is {}'.format(len(self.X_train.columns)))
            self.X_train = polynominal(self.X_train, order)
            self.X_test = polynominal(self.X_test, order)
            # print('After: the colnum of df is {}'.format(len(self.X_train.columns)))

        #  print('Train set number {}'.format(self.X_train.shape))
        if model == 'lgb':
            if self.samp_weight_control:
                self.model = LGB.LGBMClassifier()
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
                self.model = LGB.LGBMClassifier()
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
                self.model = XGB.XGBClassifier()#,use_label_encoder=False
                self.model.fit(X=self.X_train,
                               y=self.y_train,
                               sample_weight=self.train_sample_weight)
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:, 1]
                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]
            else:
                # print('without sample weight')
                self.model = XGB.XGBClassifier()#,use_label_encoder=False)
                self.model.fit(X=self.X_train,
                               y=self.y_train)
                self.train_set_predict = self.model.predict(self.X_train)
                self.train_set_predict_prob = self.model.predict_proba(self.X_train)[:, 1]
                self.test_set_predict = self.model.predict(self.X_test)
                self.test_set_predict_prob = self.model.predict_proba(self.X_test)[:, 1]

        if model == 'logreg':
            self.model = LogisticRegression()
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
            self.X_train, self.X_test = train_test_split(data, test_size=test_size, random_state=random_state)
            self.X_train = self.X_train[domain]
            self.X_test = self.X_test[domain]

            self.model = CoxPHFitter()
            self.model.fit(self.X_train, duration_col='age', event_col='death', weights_col='sampWeight',
                           cluster_col='hhid', robust=True)


class Model_non_fixed_test_size:
    def __init__(self, data, test_size, domains, model, order):
        super(Model_non_fixed_test_size, self).__init__()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop('death', axis=1),
                                                                                data['death'], test_size=test_size,
                                                                                random_state=random_state)
        self.train_sample_weight = self.X_train['sampWeight']
        self.test_sample_weight = self.X_test['sampWeight']
        self.X_train = self.X_train[domains]
        self.X_test = self.X_test[domains]

        if order != 0:
            #  print('Before: the colnum of df is {}'.format(len(self.X_train.columns)))
            self.X_train = polynominal(self.X_train, order)
            self.X_test = polynominal(self.X_test, order)
            #  print('After: the colnum of df is {}'.format(len(self.X_train.columns)))

        if model == 'lgb':
            self.model = LGB.LGBMClassifier()
            self.model.fit(X=self.X_train,
                           y=self.y_train,
                           sample_weight=self.train_sample_weight)
            self.train_set_predict = self.model.predict(self.X_train)
            self.test_set_predict = self.model.predict(self.X_test)

        if model == 'xgb':
            self.model = XGB.XGBClassifier(eval_metric='mlogloss')
            self.model.fit(X=self.X_train,
                           y=self.y_train,
                           sample_weight=self.train_sample_weight)
            self.train_set_predict = self.model.predict(self.X_train)
            self.test_set_predict = self.model.predict(self.X_test)
        '''
        if model =='logreg':
            self.model = 
            
        '''
        if model == 'cox':
            domain = list(
                set(domains + ['age', 'death', 'sampWeight', 'hhid', 'maleYN', 'blackYN', 'hispanicYN', 'otherYN',
                               'migrantYN']))
            self.X_train, self.X_test = train_test_split(data, test_size=test_size, random_state=random_state)
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
    df_coef.loc[len(df_coef),] = temp.loc[0, ]
    return df_coef

