# This file specify the parameters that will be used globally
import pandas as pd
import numpy as np
from pathlib import Path
import os
def confirm_cwd(platform):
    if platform == 'jupyter':
        os.chdir(Path.cwd().parent)
    print(f"cwd: {Path.cwd()}")

# data reader
def data_reader(source,dataset,bio):
    """
    read data 
    @param bio: whether this is a bio dataset 
    @param source: {author,us}
    @param dataset: {HRS,SHARE,CHARLES}
    @return: selected dataset 
    """

    # note that for SHARE, data_by_us.csv = recoded_data_wave_1_no_missing.csv

    data_path = Path.cwd() / f'Data/{dataset}'
    if source == 'author':
        if bio:
            df = pd.read_csv(data_path/'model_used_data/bio_all_raw_columns_no_missing.csv')
        else:
            df = pd.read_csv("/Users/valler/OneDrive - Nexus365/Replication/Python_hrsPsyMort_20190208.csv", index_col=0)
        df['ZincomeT'] = np.where(df['Zincome'] >= 1.80427, 1.80427, df['Zincome'])
        df['ZwealthT'] = np.where(df['Zwealth'] >= 3.49577, 3.49577, df['Zwealth'])

    else:

        if bio:
            df = pd.read_csv(data_path/'model_used_data/df_by_us_bio.csv')
            df['eversmokeYN'] = df['eversmokeYN'] * -1
            df.rename(columns={'deathYN': 'death'}, inplace=True)

        else:
            # df = pd.read_csv(file_path+'data_preprocess/Data/merge_data_selected_author_rows_no_missing_versioin_3.csv')
            df = pd.read_csv(data_path/'model_used_data/df_by_us.csv')
        if dataset == 'HRS':
            df = df.loc[df['age'] >= 50, ]
            df.rename(columns={'deathYear': 'death_year', 'deathMonth': 'death_month'}, inplace=True)
            df['deathYR'] = df['death_year'] + df['death_month'] / 12


    return df

def standardise(col,df):
    df[col]-=df[col].mean()
    df[col]/=df[col].std()
    return df
def read_merged_data():
    """
     merge HRS and SHARE without any treatment
    @return: the merged dataset and domain_lst (intersections)
    """

    df_HRS = data_reader(source='us', dataset='HRS', bio=False)
    df_SHARE = data_reader(source='us', dataset='SHARE', bio=False)
    df_ELSA = data_reader(source='us', dataset='SHARE', bio=False)

    # wealth pre-treat
    q = 30
    df_HRS['ZwealthT'] = pd.cut(df_HRS['ZwealthT'], bins=q, labels=False)
    df_SHARE['ZwealthT'] = pd.cut(df_SHARE['ZwealthT'], bins=q, labels=False)
    df_ELSA['ZwealthT'] = pd.cut(df_ELSA['ZwealthT'], bins=q, labels=False)

    # label data
    df_HRS['dataset'] = 0
    df_SHARE['dataset'] = 1
    df_ELSA['dataset'] = 2

    # merge without treatment
    df = pd.merge(left=df_SHARE, right=df_HRS, how='outer')
    df = pd.merge(left=df, right=df_ELSA, how='outer')

    domain_lst = list(set(df_HRS.columns).intersection(set(df_SHARE.columns)).intersection(set(df_ELSA.columns)))
    domain_lst.remove('death')
    domain_lst.remove('hhid')
    domain_lst.remove('pn')

    return df, domain_lst


domain_dict = {'demographic': ['maleYN', 'blackYN', 'hispanicYN', 'migrantYN', 'age'],
                  'child_adverse': ['sumCAE', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp',
                                    'relocate',
                                    'finhelp', 'maleYN', 'blackYN', 'hispanicYN', 'migrantYN'],
                  'adult_SES': ['rocc', 'ZwealthT', 'ZincomeT', 'everrent', 'evermedicaid', 'everfoodstamp',
                                'everunemployed', 'everfoodinsec', 'Zeduccat', 'Zrecentfindiff', 'Zneighsafety',
                                'Zneighcohesion', 'Zneighdisorder', 'maleYN', 'blackYN', 'hispanicYN',
                                'migrantYN'],
                  'behavioral': ['vigactivityYN', 'modactivityYN', 'alcoholYN', 'sleepYN', 'eversmokeYN',
                                 'currsmokeYN', 'maleYN', 'blackYN', 'hispanicYN', 'migrantYN'],
                  'adult_adverse': ['sumadultAE', 'Zmajdiscrim', 'Zdailydiscrim', 'maleYN', 'blackYN', 'hispanicYN',
                                    'migrantYN'],
                  'social_connection': ['Znegchildren', 'Znegfamily', 'Znegfriends', 'Zposchildren', 'Zposfamily',
                                        'nevermarried', 'everdivorced', 'maleYN', 'blackYN', 'hispanicYN',

                                        'migrantYN'],
                  'psych': ['Zagreeableness', 'Zangerin', 'Zangerout', 'Zanxiety', 'Zconscientiousness',
                            'Zcynhostility', 'Zextroversion', 'Zhopelessness', 'Zlifesatis', 'Zloneliness',
                            'Znegaffect', 'Zneuroticism', 'Zopenness', 'Zoptimism', 'Zperceivedconstraints',
                            'Zperceivedmastery', 'Zpessimism', 'Zposaffect', 'Zpurpose', 'Zreligiosity', 'maleYN',
                            'blackYN', 'hispanicYN', 'migrantYN'],
                  'bio': ['A1C', 'HDL', 'TC', 'CYSC_IMP', 'CRP_IMP', 'BLVERSION', 'BIOWGTR'],
                  'bio_adjusted': ['A1C_ADJ', 'HDL_ADJ', 'TC_ADJ', 'CYSC_ADJ', 'CRP_ADJ', 'BLVERSION', 'BIOWGTR'],
                  'all': ['age', 'rocc', 'fathersocc', 'Zfatherseduc', 'Zmotherseduc', 'fatherunemp', 'relocate',
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
                                       'A1C_ADJ', 'HDL_ADJ', 'TC_ADJ', 'CYSC_ADJ', 'CRP_ADJ', 'BLVERSION']
                  }
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
                "ZwealthT": "Wealth",
                "age": "Age",
                # Income here
                "ZincomeT": "Income",
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
                'A1C': 'HbA1c',
                'A1C_ADJ': 'HbA1c',
                'HDL': "HDL",
                'HDL_ADJ': "HDL",
                'TC': "Total Cholesterol",
                'TC_ADJ': "Total Cholesterol",
                'CYSC_IMP': "Cystatin C",
                'CYSC_ADJ': "Cystatin C",
                'CRP_IMP': "CRP",
                'CRP_ADJ': "CRP",
                'BLVERSION': "Bio Collected Version",
                'BIOWGTR': "Bio Weights"
                }

# model parameters
model_params = {"random_state": 87785,
                "domain_dict": domain_dict,
                "var_dict":var_dict,
                'test_size':0.3,
                'y_colname':'death'}




'''
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
