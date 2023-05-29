import xgboost as XGB
import lightgbm as LGBM
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from src import params

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


# models
def get_models():
    models = {}
    models['XGB'] = XGB.XGBClassifier()
    models['LGB'] = LGBM.LGBMClassifier()
    models['SGD'] = SGDClassifier(loss='log_loss')
    models['DecisionTree'] = DecisionTreeClassifier()
    models['AdaBoost'] = AdaBoostClassifier()
    models['LogisticRegression'] = LogisticRegression(solver='liblinear')
    models['SVC'] = SVC(gamma='scale', probability=True)
    models['Gaussian'] = GaussianNB()
    models['KNeighbors'] = KNeighborsClassifier()
    models['Bagging'] = BaggingClassifier(n_estimators=10)
    models['RandomForest'] = RandomForestClassifier(n_estimators=10)
    models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=10)
    return models


def meta_model(df_pred):
    """
    the model that ensemble the predictions from metamodels to one predicted result
    @param df_pred:
    @return:
    """
    X = df_pred[list(get_models().keys())]
    y = list(df_pred['ori_data'])

    # model = LinearRegression()
    model = LGBM.LGBMClassifier()
    model.fit(X, y)
    return model


def meta_data_generation(X, y, y_colname, k, domain_list, random_state):
    """
    generate meta data to develop meta model
    """

    models = get_models()
    kfold = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)
    df_pred = pd.DataFrame(columns=['fold', 'ori_data'] + list(models.keys()))

    # train on each subset
    for i, (train_idx, test_idx) in enumerate(kfold.split(X, y[y_colname])):

        # get the fold data
        print(f'fold {i + 1}')
        X_fold_train, y_fold_train = X.loc[train_idx,], y.loc[train_idx, y_colname]
        X_fold_test, y_fold_test = X.loc[test_idx,], y.loc[test_idx, y_colname]

        # train lerner
        temp = pd.DataFrame()

        temp['fold'] = [i + 1] * len(test_idx)
        temp['y_ind'] = test_idx
        temp['ori_data'] = list(y_fold_test)

        for model in models.keys():
            models[model].fit(X_fold_train[domain_list], y_fold_train)
            pred = [x[1] for x in models[model].predict_proba(X_fold_test[domain_list])]
            temp[f'{model}_label'] = models[model].predict(X_fold_test[domain_list])
            temp[model] = pred

        df_pred = pd.concat([df_pred, temp], axis=0)
    return df_pred


def base_model_prediction(base_models, X, y, X_test, y_test):
    """
    X,y:train dataset
    """
    df_base_pred = pd.DataFrame(columns=['ori_data'] + list(base_models.keys()))
    df_base_pred['ori_data'] = list(y_test)

    for model_name in base_models.keys():
        base_models[model_name].fit(X, y)  # train in the whole train set
        df_base_pred[model_name] = [x[1] for x in base_models[model_name].predict_proba(X_test)]
        df_base_pred[f'{model_name}_label'] = base_models[model_name].predict(X_test)
    return base_models, df_base_pred


# main class
class superlearner():

    def __init__(self, data, test_size, domain_list, y_colname, k, random_state):
        super(superlearner, self).__init__()
        self.name = 'sl'
        # first split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(y_colname, axis=1), data[y_colname], test_size=test_size, random_state=random_state)

        # step.1 prepare for train-test splitting
        self.X_train.reset_index(inplace=True)
        self.y_train = pd.DataFrame(self.y_train).reset_index()

        # step.2 meta data generating
        self.df_meta = meta_data_generation(X=self.X_train, y=self.y_train, y_colname=y_colname, k=k, domain_list=domain_list, random_state=random_state)

        # fit the metamodel (an ensemble model that bring the meta-data together)
        self.meta_model = meta_model(df_pred=self.df_meta)

        # store the predictions made by metamodels

        self.base_models, self.df_base_pred = base_model_prediction(base_models=get_models(),
                                                                    X=self.X_train[domain_list],
                                                                    X_test=self.X_test[domain_list],
                                                                    y=self.y_train['death'],
                                                                    y_test=self.y_test)

        self.meta_x = self.df_base_pred.drop(columns=['ori_data'])
        proba_columns = [x for x in self.meta_x.columns if 'label' not in x]

        self.df_base_pred['sl'] = [x[1] for x in self.meta_model.predict_proba(self.meta_x[proba_columns])]

        label_columns = [x for x in self.meta_x.columns if 'label' in x]
        meta_x_labels = self.meta_x[label_columns]

        self.df_base_pred['sl_label'] = self.meta_model.predict(meta_x_labels.rename(columns={x:x.replace('_label','') for x in label_columns}))


