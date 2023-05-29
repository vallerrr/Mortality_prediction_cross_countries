import pandas as pd
import numpy as np
from pathlib import Path
from src import params
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

def read_and_reshape_data(dataset, domain_key):
    bio=True if dataset=='HRS' else False
    df = params.data_reader(source='us', dataset=dataset, bio=bio)
    model_params = params.model_params

    # generating death age
    df['death_age'] = [age if death == 1 else None for age, death in zip(df['age'], df['death'])]
    df['death_age'].describe()

    # select only rows with death_age != None
    model_params['y_colname'] = 'death_age'
    df = df.loc[df['death_age'].notna(),]

    # remove age from the prediction
    df.drop(['age'], inplace=True, axis=1)
    if 'age' in model_params['domain_dict'][domain_key]:  model_params['domain_dict'][domain_key].remove('age')

    # only select columns that are in the domain_dict_lst
    domain_vars = list(set(model_params['domain_dict'][domain_key]).intersection(set(df.columns)))
    df = df[domain_vars + ['death_age']]
    df.dropna(axis=1, inplace=True)

    return df, model_params

def regression_evaluation(true,pred):
    mse = mean_squared_error(true, pred, squared=True)
    rmse = mean_squared_error(true, pred, squared=False)
    mae = mean_absolute_error(true, pred)

    print(f'MSE={mse}\nRMSE={rmse}\nMAE={mae}')
def train_and_predict(df,model_params,model,model_name,df_res):

    X, X_test, y, y_test = train_test_split(df.drop(['death_age'], axis=1), df['death_age'],
                                            test_size=model_params['test_size'],
                                            random_state=model_params['random_state'])
    if df_res.empty:
        df_res['true_age']=y_test

    model.fit(X, y)
    y_test_pred = model.predict(X_test)
    df_res[f'{model_name}_pred'] = y_test_pred

    return df_res


# main ---------
for dataset in ['HRS','SHARE']:
    print('\n',dataset)
    df,model_params=read_and_reshape_data(dataset,'all')
    models = {'xgb':xgb.XGBRegressor(),
              'lgb':lgb.LGBMRegressor()}

    for model_name in models.keys():
        try:
            df_res = train_and_predict(df=df, model_params=model_params, model=models[model_name], model_name=model_name, df_res=df_res)
        except:
            df_res = pd.DataFrame()
            df_res = train_and_predict(df=df, model_params=model_params, model=models[model_name], model_name=model_name, df_res=df_res)
        # evaluation
        print(f'for model {model_name}')
        regression_evaluation(df_res['true_age'], df_res[f'{model_name}_pred'])

