import pandas as pd
import os
import re
from data_preprocess.Codes.Params import recode_var_dict, var_dict,read_df
import numpy as np

cross_wave = pd.read_stata(
    "/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/CrossWaveTracker/trk2018v2a/trk2018tr_r.dta")
weight = cross_wave[['hhid', 'pn', 'kwgtr', 'lwgtr']]
cross_wave['hhidpn'] = [str(x) + str(y) for x, y in zip(cross_wave['hhid'], cross_wave['pn'])]

data_path = "/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/Variables/vars_to_merge"

# get the list of files in the folder
files_lst = [x if 'csv' in x else None for x in os.listdir(data_path)]
while None in files_lst: files_lst.remove(None)

# for non-standard variables, we have to deal with the cross-item match first




def read_file(file):
    # 1. read the processed file, either create hhidpn or only keep hhid
    # 2. create the check_mark dict, 'year' indicate the year of the source file,
    #    'household' indicates the level of data
    #    'multi_level' indicates the source file is not just from 2006/2008, but a composite file from multiple waves
    # 4. delete the rows without complete ['HHID','PN'] beforehand
    check_mark = {}
    if '06' in file:
        check_mark['year'] = 2006
    else:
        check_mark['year'] = 2008
    temp_1 = pd.read_csv(data_path + '/' + file, index_col=0)

    if len(temp_1.columns) > 3:
        check_mark['multi_wave'] = True
    else:
        check_mark['multi_wave'] = False
        temp_1.dropna(subset=['value'], inplace=True)
    if 'PN' in temp_1.columns:
        check_mark['household_mark'] = False
        temp_1.dropna(subset=['HHID', 'PN'], inplace=True)
        temp_1['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                            zip(temp_1['HHID'], temp_1['PN'])]

    else:
        temp_1.dropna(subset=['HHID'], inplace=True)
        check_mark['household_mark'] = True

    return temp_1, check_mark


def mark_the_row(temp, df, check_mark, var_name):
    # 1. match the rows in the temp and df for column var_name
    # 2. can process both individual level or household level
    # 3. it also updates the interview_year
    # 4. add the calibrate question:
    #    please only use the psychological variable to calibrate the interview year
    # 5. add function to deal with vars from multi waves and there is only one file,
    #    this can only be applied on variables of yes/no problem, allow household level data
    if check_mark['multi_wave']:
        # get information from multiple waves and only one file rather than 2 separate files
        columns_to_check = list(set(temp.columns) - {'HHID', 'hhidpn', 'PN'})
        if not check_mark['household_mark']:
            for index, row in temp.iterrows():
                hhidpn = row['hhidpn']
                nan_num = row[columns_to_check].isna().sum()

                if nan_num == len(columns_to_check):
                    df.loc[df['hhidpn'] == hhidpn, var_name] = None
                elif 1 in list(row[columns_to_check]):
                    df.loc[df['hhidpn'] == hhidpn, var_name] = 1
                else:
                    df.loc[df['hhidpn'] == hhidpn, var_name] = -1
        else:
            # at household level
            for index, row in temp.iterrows():

                hhid = row['HHID']
                nan_num = row[columns_to_check].isna().sum()
                if hhid in list(df.hhid):
                    length = len(df[df['hhid'] == hhid])

                    if nan_num == len(columns_to_check):
                        df.loc[df['hhid'] == hhid, var_name] = [None] * length
                    elif 1 in list(row[columns_to_check]):

                        df.loc[df['hhid'] == hhid, var_name] = [1] * length
                    else:
                        df.loc[df['hhid'] == hhid, var_name] = [-1] * length


    else:

        if check_mark['household_mark']:
            for index, row in temp.iterrows():
                hhid = row['HHID']
                if hhid in list(df.hhid):
                    length = len(df[df['hhid'] == hhid])
                    df.loc[df['hhid'] == hhid, var_name] = [row['value']] * length
                else:
                    continue


        else:
            cover_year = input('do you want to use this var to calibrate the interview year? y/n')
            if cover_year == 'y':
                for index, row in temp.iterrows():
                    hhidpn = row['hhidpn']
                    if hhidpn in list(df.hhidpn):
                        if np.isnan(row['value']):
                            pass
                        else:
                            df.loc[df['hhidpn'] == hhidpn, var_name] = row['value']

                            df.loc[df['hhidpn'] == hhidpn, 'interview_year'] = check_mark['year']
                    else:
                        continue
            else:
                for index, row in temp.iterrows():
                    hhidpn = row['hhidpn']
                    if hhidpn in list(df.hhidpn):
                        if np.isnan(row['value']):
                            # if the row in this file is nan, keep it's original value
                            pass
                        else:
                            df.loc[df['hhidpn'] == hhidpn, var_name] = row['value']
                    else:
                        continue

    return df


df = read_df()

var_dict = var_dict

for var_name, var_explain in var_dict.items():
    if not var_explain[1]:
        print('Start to merge {},{}'.format(var_name, var_explain[0]))
        related_files=[]
        for x in files_lst:
            if var_name in x:
                related_files.append(x)

        print('files found: {}'.format(related_files))
        input_ = input('Continue? y/s(kip)/b(reak)')
        if input_ == 'y':
            # update files
            files_lst = [x if 'csv' in x else None for x in os.listdir(data_path)]
            while None in files_lst: files_lst.remove(None)
            files = [file_name for file_name in files_lst if var_explain[0].replace(' ', '_') in file_name]
            if len(files) == 0:
                filenames = input('please input the filenames, with , as the breaker')
                files = filenames.split(',')
            print('we are going to merge file {}'.format(files))
            # read the two files from 08 or 06
            for file in files:
                print('process file {}'.format(file))
                temp, check_mark = read_file(file)
                df = mark_the_row(temp, df, check_mark, var_name)
            print('finished merging for varirable {},{}'.format(var_name, var_explain[0]))
            var_dict[var_name][1] = True
            if len(var_explain) == 3:
                continue
            else:
                var_dict[var_name].append(files)
            df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_no_standardise.csv', index=False)
        elif input_ == 's':
            continue
        else:
            temp = input('Please overwrite the params.py with the new var_dict, press random to continue')
            print(var_dict)
            break

variable_to_merge = []
for value in var_dict.values():
    if not value[1]:
        variable_to_merge.append(value[0])

# import the data created by the author and check
df_sent_by_author = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/author_data.csv')
df_sent_by_author['sampWeight'] = df_sent_by_author['sampWeight'].replace(' ', np.nan).replace('0', np.nan)
df_sent_by_author = df_sent_by_author.dropna(subset=['sampWeight'])
# df_sent_by_author.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/author_data.csv',index=False)

'''
df_sent_by_author['hhidpn']=["{:06d}".format(int(x))+'{:03d}'.format(int(y)) for x,y in zip(df_sent_by_author['hhid'],df_sent_by_author['pn'])]
# check the interview year
for index,row in df.iterrows():
    hhidpn = row['hhidpn']
    int_year= row['interview_year']
    if hhidpn in list(df_sent_by_author['hhidpn']):
        int_year_author = df_sent_by_author.loc[df_sent_by_author['hhidpn']==hhidpn,'int_year'].values
        if int_year_author == int_year:
            df.loc[index,'year_check']=True
        else:
            df.loc[index, 'year_check'] = int_year_author

df['year_check'].value_counts()


# Manual Coding Part ->do this last, to make it consistent with the interview_year
# here we draw information from the cross_wave dataset
# 0.foreign born
df['usborn'].replace({1:-1,5:1,9:None},inplace=True)

# 1. black
df['race'].replace({1:-1,2:1,0:None,7:-1,0:-1},inplace=True)
# 2. male
df['gender'].replace({1:1,2:-1},inplace=True)
# 3. hispanic

cross_wave['hispanic'].replace({5:-1,1:1,2:1,3:1,0:None},inplace=True)
df = pd.merge(left=df,right=cross_wave.loc[:,['hispanic','hhidpn']],left_on=['hhidpn'],right_on=['hhidpn'])


# 4. age
for index,row in df.iterrows():
    hhidpn=row['hhidpn']
    if row['interview_year'] == 2006:
        df.loc[index,'age'] = cross_wave.loc[cross_wave['hhidpn']==hhidpn,'kage'].values
    elif row['interview_year'] == 2008:
        df.loc[index,'age']=cross_wave.loc[cross_wave['hhidpn']==hhidpn,'lage'].values
    else:
        try:
            df.loc[index, 'age'] = cross_wave.loc[cross_wave['hhidpn'] == hhidpn, 'lage'].values  # cover with 2008 first
        except:
            continue
df['age'].replace({999:None},inplace=True)

# rename the columns that have been processed
df.rename(columns={'usborn':'migrantYN','hispanic':'hispanicYN','race':'blackYN','gender':'maleYN','exdeathmo':'deathMonth','exdeathyr':'deathYear','exdodsource':'deathReason'},inplace=True)
df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data.csv',index=False)

for value in var_dict_2.values():
    if value[1]:
        print(value[0])
df['eversmokeYN'].value_counts()
df_sent_by_author['nevermarried'].value_counts()
'''
