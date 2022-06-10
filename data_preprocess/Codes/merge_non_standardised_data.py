import pandas as pd
import os
import re
from data_preprocess.Codes.Params import recode_var_dict, var_dict,read_df
import numpy as np


# data_path = "/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/Variables/vars_to_merge"
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

    temp_1 = pd.read_csv(data_path + '/' + file)

    if len(temp_1.columns) > 4:
        check_mark['multi_wave'] = True
    else:
        check_mark['multi_wave'] = False


    if 'PN' in temp_1.columns:

        check_mark['household_mark'] = False

        df_06 = temp_1.loc[:, ['HHID', 'PN', 'value_06']]
        df_08 = temp_1.loc[:, ['HHID', 'PN', 'value_08']]

        df_06 = df_06.dropna(subset=['value_06'])
        df_08 = df_08.dropna(subset=['value_08'])

        df_06.dropna(subset=['HHID', 'PN'], inplace=True)
        df_08.dropna(subset=['HHID', 'PN'], inplace=True)

        df_06['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                            zip(df_06['HHID'], df_06['PN'])]
        df_08['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                           zip(df_08['HHID'], df_08['PN'])]

        df_06.rename(columns={'value_06': 'value'}, inplace=True)
        df_08.rename(columns={'value_08': 'value'}, inplace=True)




    else:
        df_06 = temp_1.loc[:, ['HHID','value_06']]
        df_08 = temp_1.loc[:, ['HHID', 'value_08']]

        df_06 = df_06.dropna(subset=['value_06'])
        df_08 = df_08.dropna(subset=['value_08'])

        df_06.dropna(subset=['HHID'], inplace=True)
        df_08.dropna(subset=['HHID'], inplace=True)


        df_06.rename(columns={'value_06': 'value'}, inplace=True)
        df_08.rename(columns={'value_08': 'value'}, inplace=True)

        check_mark['household_mark'] = True

    return df_06, df_08, check_mark


def mark_the_row(temp, df, check_mark, var_name):
    # 1. match the rows in the temp and df for column var_name
    # 2. can process both individual level or household level
    # 3. it also updates the interview_year
    # 4. add the calibrate question:
    #    please only use the psychological variable to calibrate the interview year
    # 5. add function to deal with vars from multi waves and there is only one file,
    #    this can only be applied on variables of yes/no problem, allow household level data


    #first we remove all values in the column

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



var_dict = {'Male': ['maleYN', True],
 'Black': ['blackYN', True],
 'Hispanic': ['hispanicYN', True],
 'Foreign Born': ['migrantYN', True],
 'Lower Education Father': ['Zfatherseduc', True],
 'Lower Education Mother': ['Zmotherseduc', True],
 'Lower Father Occupational Status': ['fathersocc', True],
 'Relocated Homes in Childhood': ['relocate', True],
 'Family Received Financial Help in Childhood': ['finhelp', True],
 'Father was Unemployed in Childhood': ['fatherunemp', True],
 'Lower Agreeableness ': ['Zagreeableness', True, []],
 'Childhood Psychosocial Adversities': ['sumCAE', True, []],
 'Lower Wealth': ['ZwealthT', False, []],
 'Age': ['age', True],
 'Lower Income': ['ZincomeT', False, []],
 'Lower Occupational Status': ['rocc', False],
 'History of Renting': ['everrent', True],
 'History of Medicaid': ['True', False],
 'History of Food Stamps': ['True', False],
 'History of Unemployment': ['everunemployed', False],
 'History of Food Insecurity': ['everfoodinsec', True],
 'Lower Education': ['Zeduccat', True],
 'Recent Financial Difficulties': ['Zrecentfindiff',
  True,
  ['19_Recent_Financial_Difficulties_20062008.csv']],
 'Lower Neighborhood Safety': ['Zneighsafety', True, []],
 'Lower Neighborhood Cohesion': ['Zneighcohesion', True, []],
 'Neighborhood Disorder': ['Zneighdisorder', False, []],
 'Low/No Vigorous Activity': ['vigactivityYN', False],
 'Low/No Moderate Activity': ['modactivityYN', False],
 'Alcohol Abuse': ['alcoholYN', False],
 'Sleep Problems': ['sleepYN', True],
 'History of Smoking': ['eversmokeYN', False],
 'Current Smoker': ['currsmokeYN', False],
 'Adulthood Psychosocial Adversity': ['sumadultAE', True, []],
 'Major Discrimination': ['Zmajdiscrim',
  True,
  ['40_Major_Discrimination.csv']],
 'Daily Discrimination': ['Zdailydiscrim',
  True,
  ['39_Daily_Discrimination.csv']],
 'Negative Interactions with Children': ['Znegchildren', True, []],
 'Negative Interactions with Family': ['Znegfamily', True, []],
 'Negative Interactions with Friends': ['Znegfriends', True, []],
 'Lower Positive Interactions with Children': ['Zposchildren', True, []],
 'Lower Positive Interactions with Family': ['Zposfamily', True, []],
 'Lower Positive Interactions with Friends': ['Zposfriends', True, []],
 'History of Divorce': ['everdivorced', False],
 'Never Married': ['nevermarried', False],
 'Anger In': ['Zangerin', True, ['25_Anger_In.csv']],
 'Anger Out': ['Zangerout', True, ['24_Anger_Out.csv']],
 'Trait Anxiety': ['Zanxiety', True, ['37_Trait_Anxiety.csv']],
 'Lower Conscientiousness': ['Zconscientiousness', True, []],
 'Cynical Hostility': ['Zcynhostility', True, ['26_Cynical_Hostility.csv']],
 'Lower Extroversion': ['Zextroversion', True, []],
 'Hopelessness': ['Zhopelessness', True, ['27_Hopelessness.csv']],
 'Lower Life Satisfaction': ['Zlifesatis', True, []],
 'Loneliness': ['Zloneliness', True, ['28_Loneliness.csv']],
 'Negative Affectivity': ['Znegaffect', True, []],
 'Lower Neuroticism': ['Zneuroticism', True, []],
 'Lower Openness to Experiences': ['Zopenness', True, []],
 'Lower Optimism': ['Zoptimism', True, []],
 'Perceptions of Obstacles': ['Zperceivedconstraints', True, []],
 'Lower Sense of Mastery': ['Zperceivedmastery', True, []],
 'Pessimism': ['Zpessimism', True, ['41_Pessimism.csv']],
 'Lower Positive Affectivity': ['Zposaffect', True, []],
 'Lower Purpose in Life': ['Zpurpose', True, []],
 'Lower Religiosity': ['Zreligiosity', True, []]}


# df = read_df()
df=pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/merge_data_non_standardise_before.csv')
df['hhidpn'] = ["{:06d}".format(int(x)) + '{:03d}'.format(int(y)) for x, y in
                            zip(df['hhid'], df['pn'])]

for var_name, var_explain in var_dict.items():
    var_col_name=var_explain[0]
    if not var_explain[1]:
        print('Start to merge {},{}'.format(var_name, var_col_name))
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
            files = [file_name for file_name in files_lst if var_name.replace(' ', '_') in file_name]
            if len(files) == 0:
                filenames = input('please input the filenames, with , as the breaker')
            else:
                filenames=files[0]
            print('we are going to merge file {}'.format(filenames))
            # read the two files from 08 or 06

            print('process file {}'.format(filenames))
            df_06,df_08, check_mark = read_file(filenames)
            df[var_col_name] = [None] * len(df[var_col_name])

            check_mark['year']= 2006
            df = mark_the_row(df_06, df, check_mark, var_col_name)

            df['Zneighdisorder'].value_counts()
            check_mark['year'] = 2008
            df = mark_the_row(df_08, df, check_mark, var_col_name)

            print('finished merging for varirable {},{}'.format(var_name, var_col_name))
            var_dict[var_name][1] = True

            # save first
            print("for var {}, uniques= {}, mean={}, max={},min={}".format(var_col_name,len(df[var_col_name].unique()),np.mean(df[var_col_name]),df[var_col_name].max(),df[var_col_name].min()))
            df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_non_standardise.csv',
                      index=False)
            # stndardise check
            standardise_check= input('do you want to standardise this var? y/n')
            if standardise_check=='y':
                df.loc[:,var_col_name]  =df.loc[:,var_col_name].astype('float')

                df.loc[:,var_col_name]= (df.loc[:,var_col_name]-np.mean(df.loc[:,var_col_name]))/np.std(df.loc[:,var_col_name])

                # this version is standardised one
                print("for var {}, uniques= {}, mean={}, max={},min={}".format(var_col_name,
                                                                               len(df[var_col_name].unique()),
                                                                               np.mean(df[var_col_name]),
                                                                               df[var_col_name].max(),
                                                                               df[var_col_name].min()))
                df.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_non_standardise_before.csv', index=False)
                print('save_standradised_version')
            # change the dict accordingly
            if len(var_explain) == 3:
                continue
            else:
                var_dict[var_name].append(files)
                # this version is not standardised one

        elif input_ == 's':
            continue
        else:
            temp = input('Please overwrite the params.py with the new var_dict, press random to continue')
            print(var_dict)
            break

'''var_name='sumCAE'
df = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/merge_data_non_standardise_before.csv')
                
print("after standardise, for var {}, uniques= {}, mean={}, max={},min={},{}".format(var_name, len(df[var_name].unique()),
                                                                               np.mean(df[var_name]),
                                                                               df[var_name].max(), df[var_name].min(),df[var_name].unique()))
'''

