import numpy as np
import pandas as pd
from pathlib import Path
from src import DataImport
import re
import sys
import random

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

# ------------------------------------------------------------------------------------------------
# Step1: read data
# ------------------------------------------------------------------------------------------------
bio_08 = pd.read_stata('/Users/valler/OneDrive - Nexus365/Dissertation/biomarker/BIOMK08BL_R.dta')
bio_06 = pd.read_stata('/Users/valler/OneDrive - Nexus365/Dissertation/biomarker/BIOMK06BL_R.dta')

df = DataImport.data_reader_by_us(bio=False)

# Step1 End Here ---------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# Step2: combine HHID and PN together for df, bio_06 and bio_08
# ------------------------------------------------------------------------------------------------
df['hhidpn']=df['hhidpn'].astype(int)
df['hhid']=df['hhid'].astype(int)

bio_06['HHID'] = bio_06['HHID'].astype(int)
bio_06['PN'] = bio_06['PN'].astype(int)

bio_08['HHID'] = bio_08['HHID'].astype(int)
bio_08['PN'] = bio_08['PN'].astype(int)

bio_06['hhidpn']=[str(bio_06.loc[index,'HHID'])+'0'+str(bio_06.loc[index,'PN']) for index in bio_06.index]
bio_08['hhidpn']=[str(bio_08.loc[index,'HHID'])+'0'+str(bio_08.loc[index,'PN']) for index in bio_08.index]

bio_06['hhidpn'] = bio_06['hhidpn'].astype(int)
bio_08['hhidpn'] = bio_08['hhidpn'].astype(int)
# Step2 End Here ---------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------
# Step3: preprocess bio_06 and bio_08
# ------------------------------------------------------------------------------------------------
# For bio_06, only keep the following columns
bio_06_columns = ['hhidpn', 'KA1CBIOS', 'KA1C_ADJ',
                  'KHDlBIOS', 'KHDl_ADJ',
                  'KTCBIOS', 'KTC_ADJ',
                  'KCYSC_IMP', 'KCYSC_ADJ',
                  'KCRP_IMP', 'KCRP_ADJ',
                  'KBlVERSION', 'KBIOWGTR']

bio_06=bio_06[bio_06_columns]

bio_06_columns = [x.replace('K','').replace('BIOS','').replace('l','L') for x in bio_06.columns]
bio_06_col_dict = {x:y for x,y in zip(list(bio_06.columns),bio_06_columns)}
bio_06=bio_06.rename(columns=bio_06_col_dict)



# construct 3 key variables for bio_08 (non-adjusted, multiple data sources)
# 1. TC, missing = 299
bio_08['TC'] = [x if np.isnan(y) else y for x,y in zip(bio_08['LTCUW'],bio_08['LTCBIOS'])]
bio_08['TC'].isna().sum()

# 2. HDL , missing = 776
bio_08['HDL'] = [x if np.isnan(y) else y for x,y in zip(bio_08['LHDLBIOS'],bio_08['LHDLUW'])]
bio_08['HDL'].isna().sum()

# 3. HbA1c , missing = 73
bio_08['A1C'] = [x if np.isnan(y) else y for x,y in zip(bio_08['LA1CBIOS'],bio_08['LA1CFLEX'])]
bio_08['A1C'].isna().sum()

bio_08_columns = ['hhidpn','A1C', 'LA1C_ADJ',
                  'HDL', 'LHDL_ADJ',
                  'TC','LTC_ADJ',
                  'LCYSC_IMP','LCYSC_ADJ',
                  'LCRP_IMP','LCRP_ADJ',
                  'LBLVERSION', 'LBIOWGTR']

bio_08 = bio_08[bio_08_columns]
bio_08_columns = [x[1:] if x.startswith('L') else x for x in bio_08.columns]
bio_08_col_dict = {x:y for x,y in zip(list(bio_08.columns),bio_08_columns)}
bio_08=bio_08.rename(columns=bio_08_col_dict)
#bio_08.columns

bio = pd.concat([bio_08,bio_06],axis=0)
# bio.to_csv(Path.cwd()/'model_used_data/bio_all.csv')
#columns: 'A1C', 'A1C_ADJ', 'HDL', 'HDL_ADJ', 'TC', 'TC_ADJ',
#'CYSC_IMP', 'CYSC_ADJ', 'CRP_IMP', 'CRP_ADJ', 'BLVERSION', 'BIOWGTR'

# ------------------------------------------------------------------------------------------------
# Step4: merge bio_06,bio_08 and df to bio_all_raw and check duplicates of hhidpn
# ------------------------------------------------------------------------------------------------

# combine 06 and 08 with df separately
combined_06 = df.merge(bio_06, left_on=['hhidpn'], right_on=['hhidpn'])
combined_08 = df.merge(bio_08, left_on=['hhidpn'], right_on=['hhidpn'])
column_difference = list(set(combined_08.columns)-set(combined_06.columns))
bio_all_raw = pd.concat([combined_08, combined_06], axis=0)


# bio_all_raw.to_csv(Path.cwd()/'model_used_data/bio_all_with_df_raw.csv')

print('bio_all_raw has {} rows and df has {} rows'.format(len(bio_all_raw),len(df)))

unique_hhidpn = list(set(bio_all_raw['hhidpn']))
print("no overlapped hhidpn in the bio_all_raw, since unique_hhidpn = bio_all_raw = {}".format(len(unique_hhidpn)))

# ------------------------------------------------------------------------------------------------
# TODO: reconstruct the sample weight?
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Step5: missing value imputation
# ------------------------------------------------------------------------------------------------


# check missing
missing_columns = []
for column in bio_all_raw.columns:
    missings = bio_all_raw[column].isna().sum()
    if (missings > 10) & ('death' not in column):
        missing_columns.append(column)
        print('for column {}, there are {} missing'.format(column, missings))

# Make an instance and perform the imputation
# first drop death related variables ['death','death_year','deathYR','death_month'],78 columns left
bio_all_raw_columns = [x if 'death' not in x else 0 for x in bio_all_raw.columns]
while 0 in bio_all_raw_columns:
    bio_all_raw_columns.remove(0)

bio_all_raw_columns_no_deaths = bio_all_raw[bio_all_raw_columns]
# bio_all_raw_columns_no_deaths.to_csv(Path.cwd()/'model_used_data/bio_all_raw_columns_no_deaths.csv')
random.seed(2022)

bio_all_raw_columns_no_deaths=pd.read_csv(Path.cwd()/'model_used_data/bio_all_raw_columns_no_deaths.csv',index_col=0)
imputer = MissForest(n_estimators=500, max_iter=5)
bio_all_raw_columns_no_deaths_no_missing = imputer.fit_transform(bio_all_raw_columns_no_deaths)
# missing values are computed at BMRC


# computation method 1: compute
no_missing = pd.DataFrame(data=bio_all_raw_columns_no_deaths_no_missing, columns=bio_all_raw_columns_no_deaths.columns)
# no_missing['index']=no_missing['index'].astype(int)

# computation method 2

no_missing[['death','death_year','deathYR','death_month','deathReason']]=df[['death','death_year','deathYR','death_month','deathReason']]
no_missing.to_csv(Path.cwd()/'model_used_data/df_by_us_bio.csv',index=False)












