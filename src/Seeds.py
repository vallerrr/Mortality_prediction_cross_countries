from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random
import lightgbm as LGB
from src import Evaluate
from src import DataImport
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from pathlib import Path


def store_coef(count, k, model_name, seed, model, domains, df_coef, cv_name):
    NOK = str(count) + '/' + str(k)
    coef_dict = {'model': model_name, 'seed': seed, 'fold_method': cv_name, 'k': k, 'NOK': NOK}
    for coef, feature in zip(model.coef_.T, domains):
        coef_dict[feature] = coef[0]
    temp = pd.DataFrame(coef_dict, index=[0])
    df_coef.loc[len(df_coef),] = temp.loc[0,]
    return df_coef





pd.set_option('precision', 9)
# read data
df = DataImport.data_reader_by_us(bio=True)
domain_dict = DataImport.domain_dict()
domains = list(set(domain_dict['all_bio']))

# control zone

model_name = 'lgb'
test_size = 0.3
seed_min_included = 1
seed_max_not_included = 3
k_list = [1, 10, 50, 100, 200, 500, 1000]

# specify dfs
df_eval = pd.DataFrame(columns=['model', 'seed', 'fold_method', 'k', 'NOK', 'auc', 'f1', 'efron_r2', 'ffc_r2'])
df_coef = pd.DataFrame(columns=['model', 'seed', 'fold_method', 'k', 'NOK'] + domains)

for seed in np.arange(seed_min_included, seed_max_not_included, 1):
    for k in k_list:
        k, seed = int(k), int(seed)
        print('seed is {} and k is {}'.format(seed, k))
        random.seed(seed)
        #  train-test split
        X, test_X, y, test_y = train_test_split(df.drop('death', axis=1), df['death'], test_size=test_size, random_state=seed)
        # X, test_X, y, test_y = train_test_split(df.drop('death', axis=1), df['death'],test_size=0.3)  # , random_state=2021)
        count = 0

        # when k = 1
        if k == 1:
            NOK = str(count) + '/' + str(k)
            if model_name == 'lgb':
                model = LGB.LGBMClassifier(random_state=seed)
                model.fit(X[domains], y)
            else:
                model = LogisticRegression(random_state=seed)
                model.fit(X[domains], y)
                df_coef = store_coef(count + 1, k, model_name, seed, model, domains, df_coef, 'NO fold splitting')


            pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]
            # TODO: for each split, the predictions should be made on the test set or the cross validation test set?
            Eval = Evaluate.evaluate_metric(test_y, pred, pred_prob, y)
            df_eval.loc[len(df_eval),] = [model, seed, 'NO fold splitting', k, NOK, Eval['auc'], Eval['f1'],
                                          Eval['efron_r2'], Eval['ffc_r2']]
        else:
            cv_dict = {'KFold': KFold(n_splits=k, shuffle=True, random_state=seed),
                       'StratifiedKFold': StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)}
            for cv_name in cv_dict.keys():
                count = 0
                cv = cv_dict[cv_name]

                for train_index, test_index in (cv.split(X, y) if cv_name == 'StratifiedKFold' else cv.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # fit model with this split
                    if model_name == 'lgb':
                        model = LGB.LGBMClassifier(random_state=seed)
                        model.fit(X_train[domains], y_train)
                        count += 1
                        NOK = str(count) + '/' + str(k)
                    else:
                        model = LogisticRegression(random_state=seed)
                        model.fit(X_train[domains], y_train)
                        count += 1
                        NOK = str(count) + '/' + str(k)

                        df_coef = store_coef(count, k, model_name, seed, model, domains, df_coef, cv_name)




                    # store coefficients


                    # store result evaluations

                    # on the whole validation set
                    '''pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]
                    # TODO: for each split, the predictions should be made on the whole test set or the cross validation test set?
                    Eval = seed_evaluate_metric(test_y, pred, pred_prob, y_train, test_X['sampWeight'])
                    df_eval.loc[len(df_eval), ] = [model, seed, cv_name, k, NOK, Eval['auc'], Eval['f1'], Eval['efron_r2'], Eval['ffc_r2']]'''

                    # on the  cross validation test set
                    pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]
                    # TODO: for each split, the predictions should be made on the whole test set or the cross validation test set?
                    Eval = Evaluate.evaluate_metric(test_y, pred, pred_prob, y_train)
                    df_eval.loc[len(df_eval),] = [model, seed, cv_name, k, NOK, Eval['auc'], Eval['f1'],
                                                  Eval['efron_r2'], Eval['ffc_r2']]

# df_eval.to_csv(Path.cwd()/'Seed/df_eval_lgbm_05052022_seed1-2.csv')


