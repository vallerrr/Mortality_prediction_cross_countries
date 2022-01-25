from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


def ffc_rsquare(true, pred, train):
    # TODO：pred here should be probs or true lables? now I'm using probs
    n = float(len(true))
    t1 = np.sum(np.power(true - pred, 2.0))
    t2 = np.sum(np.power((true - (np.sum(train) / n)), 2.0))
    return 1.0 - (t1 / t2)


def efron_rsquare(true, pred):
    # here the pred is predicted probabilities
    n = float(len(true))
    t1 = np.sum(np.power(true - pred, 2.0))
    t2 = np.sum(np.power((true - (np.sum(true) / n)), 2.0))
    return 1.0 - (t1 / t2)


def store_coef(count, k, model_name, seed, model, domains, df_coef, cv_name):
    NOK = str(count) + '/' + str(k)
    coef_dict = {'model': model_name, 'seed': seed, 'fold_method': cv_name, 'k': k, 'NOK': NOK}
    for coef, feature in zip(model.coef_.T, domains):
        coef_dict[feature] = coef[0]
    temp = pd.DataFrame(coef_dict, index=[0])
    df_coef.loc[len(df_coef),] = temp.loc[0,]
    return df_coef


def seed_evaluate_metric(true, pred_label, pred_prob, train, weight):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred_label, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)
    metric_dict['f1'] = metrics.f1_score(true, pred_label, sample_weight=weight)
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['ffc_r2'] = ffc_rsquare(true, pred_prob, train)  # TODO: pred_prob or pred?

    return metric_dict

def data_reader():
    dfMort = pd.read_csv("Python_hrsPsyMort_20190208.csv", index_col=0)
    dfMort['ZincomeT'] = np.where(dfMort['Zincome'] >= 1.80427, 1.80427, dfMort['Zincome'])
    dfMort['ZwealthT'] = np.where(dfMort['Zwealth'] >= 3.49577, 3.49577, dfMort['Zwealth'])
    return dfMort

def domain_dict():
    dict = {'all': ['rocc', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp', 'relocate', 'finhelp',
                           'sumCAE', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                           'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                           'Zneighcohesion', 'Zneighdisorder', 'vigactivityYN', 'modactivityYN', 'alcoholYN', 'sleepYN',
                           'eversmokeYN', 'currsmokeYN', 'sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim', 'Znegchildren',
                           'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily', 'nevermarried', 'everdivorced',
                           'Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness', 'Zcynhostility',
                           'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness', 'Znegaffect', 'Zneuroticism',
                           'Zopenness', 'Zoptimism', 'Zperceivedconstraints', 'Zperceivedmastery', 'Zpessimism',
                           'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN', 'blackYN', 'hispanicYN', 'otherYN',
                           'migrantYN']}
    return dict

if __name__ == '__main__':
    pd.set_option('precision', 9)
    # read data
    df = data_reader()
    domain_dict = domain_dict()
    domains = list(set(domain_dict['all']))

    # control zone
    model_name = 'logreg'
    test_size = 0.3
    seed_min_included = 1
    seed_max_not_included = 2
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
                model = LogisticRegression(random_state=seed)
                model.fit(X[domains], y, sample_weight=X['sampWeight'])
                df_coef = store_coef(count + 1, k, model_name, seed, model, domains, df_coef, 'NO fold splitting')

                pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]
                # TODO: for each split, the predictions should be made on the test set or the cross validation test set?
                Eval = seed_evaluate_metric(test_y, pred, pred_prob, y, test_X['sampWeight'])
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
                        model = LogisticRegression(random_state=seed)
                        model.fit(X_train[domains], y_train, sample_weight=X_train['sampWeight'])

                        # store coefficients
                        count += 1
                        NOK = str(count) + '/' + str(k)

                        df_coef = store_coef(count, k, model_name, seed, model, domains, df_coef, cv_name)

                        # store result evaluations
                        pred, pred_prob = model.predict(test_X[domains]), model.predict_proba(test_X[domains])[:, 1]
                        # TODO: for each split, the predictions should be made on the whole test set or the cross validation test set?
                        Eval = seed_evaluate_metric(test_y, pred, pred_prob, y_train, test_X['sampWeight'])
                        df_eval.loc[len(df_eval), ] = [model, seed, cv_name, k, NOK, Eval['auc'], Eval['f1'], Eval['efron_r2'], Eval['ffc_r2']]
