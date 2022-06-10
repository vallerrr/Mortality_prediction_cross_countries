import os
from data_preprocess.Codes.Params import recode_var_dict,var_dict_1,var_dict_2,read_df,var_dict_0
import numpy as np
import pandas as pd
import math

df = read_df()
df_rand = pd.read_stata('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/Rand/randhrs1992_2018v1.dta')
def save_df(df):
    df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data.csv',index=False)
def recode_dict(df,col):
    dict={}
    for var in df[col].value_counts().index:
        replace=input('for value {}, you want it replace with'.format(var))

        dict[var]=int(replace)
    return dict




Medicaid_cols = ['r9mrprem',  'r8mrprem', 'r7mrprem', 'r6mrprem', 'r5mrprem','r4mrprem','r3mrprem']


Medicaid = df_rand.loc[:,['hhid','pn']+Medicaid_cols]


for index,row in Medicaid.iterrows():
    nan_num = row[Medicaid_cols].isna().sum()
    if nan_num == len(Medicaid_cols):
        Medicaid.loc[index, 'value'] = None
    elif 1 in list(row[Medicaid_cols]):
        Medicaid.loc[index,'value']=1
    else:
        Medicaid.loc[index, 'value'] = -1



Medicaid['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                            zip(Medicaid['hhid'], Medicaid['pn'])]


medicaid = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/Variables/10_History_of_Medicaid.csv',index_col=0)
columns = list(medicaid.columns)
columns.remove('HHID')
columns.remove('PN')

for index,row in medicaid.iterrows():
    nan_num = row[columns].isna().sum()
    if nan_num == len(columns):
        medicaid.loc[index, 'value'] = None
    elif 1 in list(row[columns]):
        medicaid.loc[index,'value']=1
    else:
        medicaid.loc[index, 'value'] = -1

medicaid.dropna(subset=['HHID', 'PN'], inplace=True)
medicaid['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                            zip(medicaid['HHID'], medicaid['PN'])]



for index,row in df.iterrows():
    hhidpn=row['hhidpn']


    if hhidpn in list(Medicaid.hhidpn):
        Medi_value=Medicaid.loc[Medicaid['hhidpn']==hhidpn,'value'].values[0]
    else:
        Medi_value=np.nan
    if hhidpn in list(medicaid.hhidpn):
        medi_value = medicaid.loc[medicaid['hhidpn'] == hhidpn, 'value'].values[0]
    else:
        medi_value=np.nan

    if (np.isnan(Medi_value)) & (np.isnan(medi_value)):
        df.loc[index,'evermedicaid'] = None
    elif ~np.isnan(medi_value):
        df.loc[index, 'evermedicaid'] = medi_value
    else:
        df.loc[index, 'evermedicaid'] = Medi_value





df['evermedicaid'].isnull().sum()

df['evermedicaid'].value_counts()

save_df(df)
