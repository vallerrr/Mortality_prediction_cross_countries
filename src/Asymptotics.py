import Models
import DataImport
import Evaluate
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def asymptotics(model_set, start, end, step, domains):
    Eva_fixed_test_size = pd.DataFrame(
        columns=['model', 'train_subset_size', 'train_auc_score', 'test_auc_score', 'train_f1_score', 'test_f1_score',
                 'train_efron_r2', 'test_efron_r2', 'train_ffc_r2', 'test_ffc_r2'])

    for model in model_set:
        print('Model: {}'.format(model))
        for train_subset_size in np.arange(start, end, step):
            print('train set size = {}'.format(train_subset_size))
            lgb = Models.Model_fixed_test_size(df, 0.2, domains, model, train_subset_size, polynomial)
            Eva_fixed_test_size.loc[len(Eva_fixed_test_size)] = [model,
                                                                 len(lgb.X_train),

                                                                 # AUC score
                                                                 Evaluate.metric(lgb).train_roc_auc_score,
                                                                 Evaluate.metric(lgb).valid_roc_auc_score,

                                                                 # F1 score
                                                                 Evaluate.metric(lgb).train_f1_score,
                                                                 Evaluate.metric(lgb).valid_f1_score,
                                                                 # Evaluate.pseudo_rmse(lgb.test_set_predict, lgb.y_test, lgb.y_train),

                                                                 # Efron_r2
                                                                 Evaluate.efron_rsquare(lgb.y_train,
                                                                                        lgb.train_set_predict_prob),
                                                                 Evaluate.efron_rsquare(lgb.y_test,
                                                                                        lgb.test_set_predict_prob),
                                                                 # ffc_r2
                                                                 Evaluate.ffc_rsquare(lgb.y_train,
                                                                                      lgb.train_set_predict_prob,
                                                                                      lgb.y_train),
                                                                 Evaluate.ffc_rsquare(lgb.y_test,
                                                                                      lgb.test_set_predict_prob,
                                                                                      lgb.y_train)]
    return Eva_fixed_test_size

random.seed(52194)
pd.set_option('precision', 9)

# read data
polynomial = 0

df = DataImport.data_reader()
domain_dict = DataImport.domain_dict()
domains = list(set(domain_dict['all']))

domain_dict.keys()
# TODO: Add Domain

model_set = ['xgb']  # 'xgb', 'logreg']

Eva_fixed_test_size = asymptotics(model_set=model_set, start=0.3, end=0.99, step=0.005, domains=domains)
Eva_fixed_test_size_new = asymptotics(model_set=model_set, start=0.9, end=1, step=0.005, domains=domains)
# auc score
fig, ax = plt.subplots(figsize=(8, 6))
for label, group in Eva_fixed_test_size.groupby('model'):
    group.plot(x='train_subset_size', y='test_auc_score', ax=ax, label=label)
plt.title('auc score {} order'.format(polynomial))
plt.show()

# f1 score


fig, ax = plt.subplots(figsize=(8, 6))
for label, group in Eva_fixed_test_size.groupby('model'):
    group.plot(x='train_subset_size', y='test_f1_score', ax=ax, label=label)
plt.title('f1 score {} order'.format(polynomial))
plt.show()

# efron_r2
fig, ax = plt.subplots(figsize=(8, 6))
for label, group in Eva_fixed_test_size.groupby('model'):
    group.plot(x='train_subset_size', y='test_efron_r2', ax=ax, label=label)
plt.title('pseudo efron r2 {} order'.format(polynomial))
plt.show()

# ffc_r2
fig, ax = plt.subplots(figsize=(8, 6))
for label, group in Eva_fixed_test_size.groupby('model'):
    group.plot(x='train_subset_size', y='test_ffc_r2', ax=ax, label=label)
plt.title('pseudo ffc r2 {} order'.format(polynomial))
plt.show()

