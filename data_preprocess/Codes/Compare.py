
import pandas as pd
import numpy as np

import sys
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest


# import the data created by the author and check
df_sent_by_author = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/author_data.csv')
df_sent_by_author['sampWeight']=df_sent_by_author['sampWeight'].replace(' ',np.nan).replace('0',np.nan)
df_sent_by_author = df_sent_by_author.dropna(subset=['sampWeight'])
# df_sent_by_author.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/author_data.csv',index=False)

# import the dataset created by us
# df = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data.csv')


df = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/merge_data_step_3.csv')
#df = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data.csv')





# only keep records that are interviewed in 2006/2008
df.dropna(subset=['interview_year'],inplace=True)  # 19193 samples left
# check the deaths first
df['death'] = [0 if np.isnan(x) else 1 for x in df['deathYear']]

# df_only_rows_in_authors = df.loc[df['year_check']=='True',]

df=df.dropna(thresh=50)


count =0
df_to_check = data_non_missing_version_2
#pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_selected_author_rows_no_missing_versioin_2.csv')
df_to_check=df_to_check.loc[df_to_check['age']>=52,]


null_records=pd.DataFrame(columns=['column','dataset_mark','max','min','mean','uniques','null'])


df_to_check=df_to_check.rename(columns={'death':'deathYN','deathYear':'death_year','deathMonth':'death_month','ZwealthT':'Zwealth', 'ZincomeT':'Zincome'})
for column in df_sent_by_author.columns:
    if column in list(df_to_check.columns):

        author_column  = df_sent_by_author.loc[:,column].replace({' ':None,'True':1,'False':0}).astype(float)
        null_records = null_records.append({'column':column, 'dataset_mark':'author', 'max':max(author_column), 'min':min(author_column), 'mean':np.mean(author_column),'uniques':len(author_column.unique()), 'null':author_column.isnull().sum()},
                                           ignore_index=True)

        count+=1
    else:
        continue

    our_column=df_to_check[column]
    our_column = our_column.replace({' ': None, 'True': 1, 'False': 0}).astype(float)
    null_records=null_records.append({'column':column,'dataset_mark':'us','max':max(our_column),'min':min(our_column), 'mean':np.mean(our_column),'uniques':len(our_column.unique()),'null':our_column.isnull().sum()},ignore_index=True)

# null_records.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/null_check.csv',index=False)


# df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_selected_int_year.csv',index=False)


# Missforest



# first version: 1. only rows iin the authors dataset
df_to_missing_forest=df_only_rows_in_authors.drop(columns=['death','deathYear','deathMonth','deathReason','year_check'])

imputer = MissForest(n_estimators=500, max_iter=5)
dfLBImp = imputer.fit_transform(df_to_missing_forest)


data_non_missing_version_1 = pd.DataFrame(columns=df_to_missing_forest.columns,data=dfLBImp)
data_non_missing_version_1.loc[:,['death','deathYear','deathMonth','deathReason']]=df_only_rows_in_authors.loc[:,['death','deathYear','deathMonth','deathReason']]\

data_non_missing_version_1.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_selected_author_rows_no_missing_versioin_1.csv',index=False)

# first version: 2. only rows have more than 20%
df_to_missing_forest=df.drop(columns=['death','deathYear','deathMonth','deathReason','year_check'])

df_to_missing_forest= df_to_missing_forest.dropna(thresh=50)
imputer = MissForest(n_estimators=500, max_iter=5)
dfLBImp = imputer.fit_transform(df_to_missing_forest)


data_non_missing_version_2 = pd.DataFrame(columns=df_to_missing_forest.columns,data=dfLBImp)
data_non_missing_version_2.loc[:,['death','deathYear','deathMonth','deathReason']]=df.loc[:,['death','deathYear','deathMonth','deathReason']]\

data_non_missing_version_2.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_selected_author_rows_no_missing_versioin_3.csv',index=False)



# lst version: with biomarkers

df=pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/merge_data_step_3.csv')

dfMort = df.loc[df['age'] >= 52,]
dfMort= dfMort.dropna(thresh=50)
df_bio = pd.read_csv('/Bio_data/bio_all.csv', index_col=0)
bio_columns = list(df_bio.columns)
bio_columns.remove('hhidpn')
bio_columns.remove('BLVERSION')


for index, row in df_bio.iterrows():
    hhidpn = float(row['hhidpn'])
    if hhidpn in list(dfMort.hhidpn):
        for column in bio_columns:
            dfMort.loc[dfMort['hhidpn'] == hhidpn, column] = row[column]


dfMort=dfMort.rename(columns={'deathYear':'death_year','deathMonth':'death_month','ZwealthT':'Zwealth', 'ZincomeT':'Zincome'})
dfMort['deathYR']=dfMort['death_year']+dfMort['death_month']/12
dfMort.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/Bio_data/df_by_us_bio.csv', index=False)

df_to_missing_forest=dfMort.drop(columns=['death','death_year','death_month','deathReason','deathYR','year_check'])

imputer = MissForest(n_estimators=500, max_iter=5)
dfLBImp = imputer.fit_transform(df_to_missing_forest)


data_non_missing_version_3 = pd.DataFrame(columns=df_to_missing_forest.columns,data=dfLBImp)
data_non_missing_version_3.loc[:,['death','death_year','death_month','deathReason','deathYR']]=dfMort.loc[:,['death','death_year','death_month','deathReason','deathYR']]

data_non_missing_version_3.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/Bio_data/df_by_us_bio.csv',index=False)
