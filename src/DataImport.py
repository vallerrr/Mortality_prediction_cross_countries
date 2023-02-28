import pandas as pd
import numpy as np
import re


file_path = "/Users/valler/Python/OX_Thesis/OX_thesis/Data/HRS/"
def data_reader(bio):
    if bio == True:
        dfMort = pd.read_csv(file_path+'model_used_data/bio_all_raw_columns_no_missing.csv')
    else:
        dfMort = pd.read_csv("/Users/valler/OneDrive - Nexus365/Replication/Python_hrsPsyMort_20190208.csv", index_col=0)
    dfMort['ZincomeT'] = np.where(dfMort['Zincome'] >= 1.80427, 1.80427, dfMort['Zincome'])
    dfMort['ZwealthT'] = np.where(dfMort['Zwealth'] >= 3.49577, 3.49577, dfMort['Zwealth'])

    return dfMort


def data_reader_by_us(bio):
    binary_treat_control = False
    if bio == True:
        dfMort = pd.read_csv(file_path+'model_used_data/df_by_us_bio.csv')
        dfMort['eversmokeYN'] = dfMort['eversmokeYN'] * -1
        dfMort.rename(columns={'deathYN':'death'}, inplace=True)

        # dfMort['ZincomeT'] = np.where(dfMort['Zincome'] >= 1.80427, 1.80427, dfMort['Zincome'])
        # dfMort['ZwealthT'] = np.where(dfMort['Zwealth'] >= 3.49577, 3.49577, dfMort['Zwealth'])

    else:

        # dfMort = pd.read_csv(file_path+'data_preprocess/Data/merge_data_selected_author_rows_no_missing_versioin_3.csv')
        dfMort = pd.read_csv(file_path+'model_used_data/df_by_us.csv')


    dfMort.rename(columns={'deathYear': 'death_year', 'deathMonth': 'death_month'}, inplace=True)
    dfMort['deathYR'] = dfMort['death_year'] + dfMort['death_month'] / 12
    return dfMort


def domain_dict():
    domain_diction = {'demographic': ['maleYN', 'blackYN', 'hispanicYN', 'migrantYN','age'],
                      'child_adverse': ['sumCAE', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp',
                                        'relocate',
                                        'finhelp', 'maleYN', 'blackYN', 'hispanicYN',  'migrantYN'],
                      'adult_SES': ['rocc', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                                    'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                                    'Zneighcohesion', 'Zneighdisorder', 'maleYN', 'blackYN', 'hispanicYN', 
                                    'migrantYN'],
                      'behavioral': ['vigactivityYN', 'modactivityYN', 'alcoholYN', 'sleepYN', 'eversmokeYN',
                                     'currsmokeYN', 'maleYN', 'blackYN', 'hispanicYN',  'migrantYN'],
                      'adult_adverse': ['sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim', 'maleYN', 'blackYN', 'hispanicYN',
                                         'migrantYN'],
                      'social_connection': ['Znegchildren', 'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily',
                                            'nevermarried', 'everdivorced', 'maleYN', 'blackYN', 'hispanicYN',
                                            
                                            'migrantYN'],
                      'psych': ['Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness',
                                'Zcynhostility', 'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness',
                                'Znegaffect', 'Zneuroticism', 'Zopenness', 'Zoptimism', 'Zperceivedconstraints',
                                'Zperceivedmastery', 'Zpessimism', 'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN',
                                'blackYN', 'hispanicYN',  'migrantYN'],
                      'bio': ['A1C',  'HDL',  'TC', 'CYSC_IMP', 'CRP_IMP',  'BLVERSION', 'BIOWGTR'],
                      'bio_adjusted': ['A1C_ADJ','HDL_ADJ','TC_ADJ', 'CYSC_ADJ', 'CRP_ADJ', 'BLVERSION', 'BIOWGTR'],
                      'all': ['age','rocc', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp', 'relocate',
                              'finhelp',
                              'sumCAE', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                              'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                              'Zneighcohesion', 'Zneighdisorder', 'vigactivityYN', 'modactivityYN', 'alcoholYN',
                              'sleepYN',
                              'eversmokeYN', 'currsmokeYN', 'sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim',
                              'Znegchildren',
                              'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily', 'nevermarried', 'everdivorced',
                              'Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness',
                              'Zcynhostility',
                              'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness', 'Znegaffect',
                              'Zneuroticism',
                              'Zopenness', 'Zoptimism', 'Zperceivedconstraints', 'Zperceivedmastery', 'Zpessimism',
                              'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN', 'blackYN', 'hispanicYN', 
                              'migrantYN'],

                      'all_bio': ['age', 'rocc', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp', 'relocate',
                              'finhelp',
                              'sumCAE', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                              'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                              'Zneighcohesion', 'Zneighdisorder', 'vigactivityYN', 'modactivityYN', 'alcoholYN',
                              'sleepYN',
                              'eversmokeYN', 'currsmokeYN', 'sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim',
                              'Znegchildren',
                              'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily', 'nevermarried', 'everdivorced',
                              'Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness',
                              'Zcynhostility',
                              'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness', 'Znegaffect',
                              'Zneuroticism',
                              'Zopenness', 'Zoptimism', 'Zperceivedconstraints', 'Zperceivedmastery', 'Zpessimism',
                              'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN', 'blackYN', 'hispanicYN', 
                              'migrantYN',
                              'A1C', 'HDL', 'TC', 'CYSC_IMP', 'CRP_IMP', 'BLVERSION'],

                      'all_bio_adjusted': ['age', 'rocc', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp', 'relocate',
                              'finhelp',
                              'sumCAE', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                              'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                              'Zneighcohesion', 'Zneighdisorder', 'vigactivityYN', 'modactivityYN', 'alcoholYN',
                              'sleepYN',
                              'eversmokeYN', 'currsmokeYN', 'sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim',
                              'Znegchildren',
                              'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily', 'nevermarried', 'everdivorced',
                              'Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness',
                              'Zcynhostility',
                              'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness', 'Znegaffect',
                              'Zneuroticism',
                              'Zopenness', 'Zoptimism', 'Zperceivedconstraints', 'Zperceivedmastery', 'Zpessimism',
                              'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN', 'blackYN', 'hispanicYN', 
                              'migrantYN',
                               'A1C_ADJ','HDL_ADJ','TC_ADJ', 'CYSC_ADJ', 'CRP_ADJ', 'BLVERSION']
                      }
    return domain_diction


def variable_dict():
    var_dict = {"maleYN": "Male",
                "blackYN": "Black",
                "hispanicYN": "Hispanic",
                "otherYN": "Other races",
                "migrantYN": "Foreign Born",
                "Zfatherseduc": "Lower Education Father",
                "Zmotherseduc": "Lower Education Mother",
                "fathersocc": "Lower Father Occupational Status",
                "relocate": "Relocated Homes in Childhood",
                "finhelp": "Family Received Financial Help in Childhood",
                "fatherunemp": "Father was Unemployed in Childhood",
                "sumCAE": "Childhood Psychosocial Adversities",
                "ZwealthT": "Lower Wealth",
                "age": "Age",
                #Income here
                "ZincomeT": "Lower Income",
                "rocc": "Lower Occupational Status",
                "everrent": "History of Renting",
                "evermedicaid": "History of Medicaid",
                "everfoodstamp": "History of Food Stamps",
                "everunemployed": "History of Unemployment",
                "everfoodinsec": "History of Food Insecurity",
                "Zeduccat": "Lower Education",
                "Zrecentfindiff": "Recent Financial Difficulties",
                "Zneighsafety": "Lower Neighborhood Safety",
                "Zneighcohesion": "Lower Neighborhood Cohesion",
                "Zneighdisorder": "Neighborhood Disorder",
                "vigactivityYN": "Low/No Vigorous Activity",
                "modactivityYN": "Low/No Moderate Activity",
                "alcoholYN": "Alcohol Abuse",
                "sleepYN": "Sleep Problems",
                "eversmokeYN": "History of Smoking",
                "currsmokeYN": "Current Smoker",
                "sumadultAE": "Adulthood Psychosocial Adversity",
                "Zmajdiscrim": "Major Discrimination",
                "Zdailydiscrim": "Daily Discrimination",
                "Znegchildren": "Negative Interactions with Children",
                "Znegfamily": "Negative Interactions with Family",
                "Znegfriends": "Negative Interactions with Friends",
                "Zposchildren": "Lower Positive Interactions with Children",
                "Zposfamily": "Lower Positive Interactions with Family",
                "Zposfriends": "Lower Positive Interactions with Friends",
                "everdivorced": "History of Divorce",
                "nevermarried": "Never Married",
                "Zagreeableness": "Lower Agreeableness ",
                "Zangerin": "Anger In",
                "Zangerout": "Anger Out",
                "Zanxiety": "Trait Anxiety",
                "Zconscientiousness": "Lower Conscientiousness",
                "Zcynhostility": "Cynical Hostility",
                "Zextroversion": "Lower Extroversion",
                "Zhopelessness": "Hopelessness",
                "Zlifesatis": "Lower Life Satisfaction",
                "Zloneliness": "Loneliness",
                "Znegaffect": "Negative Affectivity",
                "Zneuroticism": "Lower Neuroticism",
                "Zopenness": "Lower Openness to Experiences",
                "Zoptimism": "Lower Optimism",
                "Zperceivedconstraints": "Perceptions of Obstacles",
                "Zperceivedmastery": "Lower Sense of Mastery",
                "Zpessimism": "Pessimism",
                "Zposaffect": "Lower Positive Affectivity",
                "Zpurpose": "Lower Purpose in Life",
                "Zreligiosity": "Lower Religiosity",
                'A1C':'HbA1c',
                'A1C_ADJ':'HbA1c',
                'HDL':"HDL",
                'HDL_ADJ':"HDL",
                'TC': "Total Cholesterol",
                'TC_ADJ' : "Total Cholesterol",
                'CYSC_IMP': "Cystatin C",
                'CYSC_ADJ': "Cystatin C",
                'CRP_IMP': "CRP",
                'CRP_ADJ': "CRP",
                'BLVERSION': "Bio Collected Version",
                'BIOWGTR': "Bio Weights"
                }
    return var_dict


def recode_categorical_vars(column, cat_num, df):
    # pair of values and their number of observations
    cat_val_count_lst = []
    count = 0
    for index, value in df[column].value_counts().items():
        count += 1
        if count >= cat_num + 1:
            break
        else:
            # print(index,value)
            cat_val_count_lst.append([index, value])
    # cut off calculation
    cut_point = {}
    for index in range(0, cat_num - 1):
        cat_diff = cat_val_count_lst[index + 1][0] - cat_val_count_lst[index][0]
        gravity = cat_val_count_lst[index][1] / (cat_val_count_lst[index][1] + cat_val_count_lst[index + 1][1])
        cut_point[cat_val_count_lst[index][0]] = cat_val_count_lst[index][0] + gravity * cat_diff
    # cut bins
    df[column] = pd.cut(df[column], bins=[cats[0] - 100] + cats + [cats[len(cats) - 1] + 100], labels=[x[0] for x in cat_val_count_lst])
    return df
'''
# bio match
df_bio = pd.read_csv('/Users/valler/Python/OX_Thesis/OX_thesis/model_used_data/bio_all.csv',index_col=0)
bio_columns = list(df_bio.columns)
bio_columns.remove('hhidpn')
df_by_us=data_reader_by_us(bio=False)


for index,row in df_bio.iterrows():
    hhidpn=float(row['hhidpn'])
    if hhidpn in list(df_by_us.hhidpn):
        for column in bio_columns:
            df_by_us.loc[df_by_us['hhidpn']==hhidpn,column] = row[column]
            
df_by_us.to_csv('/Users/valler/Python/OX_Thesis/OX_thesis/model_used_data/df_by_us_bio.csv',index=False)

'''
