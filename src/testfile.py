"""
# Created by valler at 24/01/2025
Feature: 

"""
from src.params import data_reader
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score

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
                "Zfatherseduc": "Father Education ",
                "Zmotherseduc": "Mother Education ",
                "fathersocc": "Father Occupational Status",
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

def read_merged_data(type):
    df_HRS = data_reader(source='us', dataset='HRS', bio=False, merge=True)
    for col in df_HRS.columns:
        df_HRS[col] = df_HRS[col].astype(float)

    df_SHARE = data_reader(source='us', dataset='SHARE', bio=False, merge=True)
    df_SHARE = df_SHARE.loc[(df_SHARE['deathY'] > 2005) | (df_SHARE['death'] == 0),]
    for col in df_SHARE.columns:
        if col not in ['mergeid', 'isocountry']:
            df_SHARE[col] = df_SHARE[col].astype(float)

    df_ELSA = data_reader(source='us', dataset='ELSA', bio=False, merge=True)
    for col in df_ELSA.columns:
        df_ELSA[col] = df_ELSA[col].astype(float)
    # wealth pre-treat
    q = 30
    var = 'ZwealthT'
    df_HRS[var] = pd.cut(df_HRS[var], bins=q, labels=False)
    df_SHARE[var] = pd.cut(df_SHARE[var], bins=q, labels=False)
    df_ELSA[var] = pd.cut(df_ELSA[var], bins=q, labels=False)

    # income pre-treat
    q = 10
    var = 'ZincomeT'
    df_HRS[var] = pd.cut(df_HRS[var], bins=q, labels=False)
    # df_SHARE[var] = pd.cut(df_SHARE[var], bins=q, labels=False)
    df_ELSA[var] = pd.cut(df_ELSA[var], bins=q, labels=False)

    q = 5
    for var in ['Zfatherseduc', 'Zmotherseduc', 'Zeduccat']:
        df_HRS[var] = pd.cut(df_HRS[var], bins=q, labels=False)
        df_SHARE[var] = pd.cut(df_SHARE[var], bins=q, labels=False)
        #df_ELSA[var] = pd.cut(df_ELSA[var], bins=q, labels=False)


    unique_bins_HRS={'Zagreeableness': 5, 'sumCAE': 4, 'rocc': 6, 'Zrecentfindiff': 5, 'Zneighsafety': 5, 'Zneighcohesion': 7, 'Zneighdisorder': 7, 'Zmajdiscrim': 5, 'Zdailydiscrim': 5, 'Znegchildren': 5, 'Znegfamily': 5, 'Znegfriends': 5, 'Zposchildren': 5, 'Zposfamily': 5, 'Zposfriends': 5, 'Zangerin': 5, 'Zangerout': 5, 'Zanxiety': 5, 'Zconscientiousness': 5, 'Zcynhostility': 6, 'Zextroversion': 5, 'Zhopelessness': 6, 'Zlifesatis': 7, 'Zloneliness': 5, 'Znegaffect': 10, 'Zneuroticism': 5, 'Zopenness': 5, 'Zoptimism': 6, 'Zperceivedconstraints': 6, 'Zperceivedmastery': 6, 'Zpessimism': 6, 'Zposaffect': 10, 'Zpurpose': 6, 'Zreligiosity': 6, 'ZwealthT': 30, 'sumadultAE': 6, 'fathersocc': 6}
    for var in unique_bins_HRS.keys():
        df_HRS[var] = pd.cut(df_HRS[var], bins=unique_bins_HRS[var], labels=False)
        if var in df_SHARE.columns:
            df_SHARE[var] = pd.cut(df_SHARE[var], bins=unique_bins_HRS[var], labels=False)
        if var in df_ELSA.columns:
            df_ELSA[var] = pd.cut(df_ELSA[var], bins=unique_bins_HRS[var], labels=False)

    # label data
    df_HRS['dataset'] = 0
    df_SHARE['dataset'] = 1
    df_ELSA['dataset'] = 2


    if type == 1:
        # 1: HRS + SHARE + ELSA
        # merge without treatment
        df = pd.merge(left=df_SHARE, right=df_HRS, how='outer')
        df = pd.merge(left=df, right=df_ELSA, how='outer')

        ELSA_columns = list(set(domain_dict['all']).intersection(set(df_ELSA.columns)))
        HRS_columns = list(set(domain_dict['all']).intersection(set(df_HRS.columns)))
        SHARE_columns = list(set(domain_dict['all']).intersection(set(df_SHARE.columns)))
        domain_lst = list(set(HRS_columns).intersection(set(SHARE_columns)).intersection(set(ELSA_columns)))

    elif type == 2:
        # 2. HRS+SHARE
        df = pd.merge(left=df_SHARE, right=df_HRS, how='outer')

        HRS_columns = list(set(domain_dict['all']).intersection(set(df_HRS.columns)))
        SHARE_columns = list(set(domain_dict['all']).intersection(set(df_SHARE.columns)))

        domain_lst = list(set(HRS_columns).intersection(set(SHARE_columns)))

    elif type == 3:
        # 3. HRS+ELSA
        df = pd.merge(left=df_ELSA, right=df_HRS, how='outer')

        ELSA_columns = list(set(domain_dict['all']).intersection(set(df_ELSA.columns)))
        HRS_columns = list(set(domain_dict['all']).intersection(set(df_HRS.columns)))

        domain_lst = list(set(HRS_columns).intersection(set(ELSA_columns)))
    else:
        # 4. SHARE + ELSA
        df = pd.merge(left=df_SHARE, right=df_ELSA, how='outer')

        ELSA_columns = list(set(domain_dict['all']).intersection(set(df_ELSA.columns)))
        SHARE_columns = list(set(domain_dict['all']).intersection(set(df_SHARE.columns)))

        domain_lst = list(set(ELSA_columns).intersection(set(SHARE_columns)))

    for column in ['death', 'pn', 'hhid']:
        if column in domain_lst:
            domain_lst.remove(column)
    domain_lst.sort()
    return df, domain_lst
def ELSA_additional_evals(true_vals, pred_probs, threshold=0.5):
    # Convert probabilities to binary predictions based on the threshold
    pred_vals = [1 if prob >= threshold else 0 for prob in pred_probs]

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_vals, pred_vals).ravel()

    # Calculate metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
    precision = precision_score(true_vals, pred_vals, zero_division=0)  # Precision

    # Likelihood ratio (LR+): TPR / FPR
    lr = tpr / fpr if fpr > 0 else float('inf')  # Avoid division by zero

    return {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp, 'lr':lr}


# test performance

from src import Models
from pathlib import Path
from src.Evaluate import evaluate_metric
from src.Evaluate import metric,print_model_fits,sl_eva
from src import params
def print_evaluate_metric(metric):
    print(f'imv={metric["imv"]}\nroc-auc={metric["roc_auc"]}\npr-auc={metric["auc"]}\nf1={metric["f1"]}\nfb={metric["fb"]}\nefron_r2={metric["efron_r2"]}\npr_no_skill={metric["pr_no_skill"]}\n ')

df, domain_lst = read_merged_data(type=3)
model_params = params.model_params
model_params['random_state'] = 87785
domain_name = 'combine_all'
domain_lst.sort()

model_params['domain_dict'][domain_name]=domain_lst

model = Models.Model_fixed_test_size(data=df, model_params=model_params, domain=domain_name, model='lgb', train_subset_size=1, order=0)
print_model_fits(evas=metric(model))

#save the idauniq

all_elsa_id_in_comb = df.loc[model.X_test.index].loc[df['dataset']==2,'idauniq'].values

all_elsa_index_in_comb = df.loc[model.X_test.index].loc[df['dataset']==2,].index
all_elsa_pred = model.model.predict(model.X_test.loc[all_elsa_index_in_comb,domain_lst])
all_elsa_pred_prob = model.model.predict_proba(model.X_test.loc[all_elsa_index_in_comb,domain_lst])[:,1]
all_elsa_true = model.y_test.loc[all_elsa_index_in_comb]

ELSA_additional_evals(all_elsa_true,all_elsa_pred_prob,threshold=0.5)
print_evaluate_metric(evaluate_metric(all_elsa_true, all_elsa_pred, all_elsa_pred_prob, model.y_train))
# elsa model

df_elsa = params.data_reader(dataset='ELSA', source='us', bio=False)
domain_name = 'elsa_all'
elsa_model_params=params.model_params
elsa_model_params['random_state'] = 87785
elsa_model_params['domain_dict'][domain_name] = list(set(elsa_model_params['domain_dict']['all']).intersection(set(df_elsa.columns)))

elsa_model_params['domain_dict'][domain_name].sort()

elsa_model = Models.Model_fixed_test_size(data=df_elsa, model_params=elsa_model_params, domain=domain_name, model='lgb', train_subset_size=1, order=0)
print_model_fits(evas=metric(elsa_model))
ELSA_additional_evals(elsa_model.y_test,elsa_model.test_set_predict_prob,threshold=0.5)

elsa_id_in_test = df_elsa.loc[elsa_model.X_test.index,'idauniq'].values
intersection_id = list(set(all_elsa_id_in_comb).intersection(set(elsa_id_in_test)))
intersection_index_in_elsa = df_elsa.loc[df_elsa['idauniq'].isin(intersection_id)].index

elsa_pred = elsa_model.model.predict(elsa_model.X_test.loc[intersection_index_in_elsa,elsa_model_params['domain_dict'][domain_name]])
elsa_pred_prob = elsa_model.model.predict_proba(elsa_model.X_test.loc[intersection_index_in_elsa,elsa_model_params['domain_dict'][domain_name]])[:,1]
elsa_true = elsa_model.y_test.loc[intersection_index_in_elsa]

ELSA_additional_evals(elsa_true,elsa_pred_prob,threshold=0.5)
print_evaluate_metric(evaluate_metric(elsa_true, elsa_pred, elsa_pred_prob, elsa_model.y_train))

intersection_index_in_comb = df.loc[df['idauniq'].isin(intersection_id)].index
intersection_pred = model.model.predict(model.X_test.loc[intersection_index_in_comb,domain_lst])
intersection_pred_prob = model.model.predict_proba(model.X_test.loc[intersection_index_in_comb,domain_lst])[:,1]
intersection_true = model.y_test.loc[intersection_index_in_comb]

ELSA_additional_evals(intersection_true,intersection_pred_prob,threshold=0.5)
print_evaluate_metric(evaluate_metric(intersection_true, intersection_pred, intersection_pred_prob, model.y_train))










'''
for seed_ind in range(10):
    ELSA_seed = ELSA_seeds[seed_ind]
    COMB_seed = comb_seeds[seed_ind]

    #bolack 1 ELSA





    # block 1 ELSA
    method = 'ELSA model'
    df = params.data_reader(dataset='ELSA', source='us', bio=False)
    domain_name = 'elsa_all'
    elsa_model_params = params.model_params
    elsa_model_params['random_state'] = ELSA_seed
    # print('ELSA SEEDS',elsa_model_params['random_state'])
    elsa_model_params['domain_dict'][domain_name] = list(set(elsa_model_params['domain_dict']['all']).intersection(set(df.columns)))
    elsa_model_params['domain_dict'][domain_name].sort()
    # recode a new domain dict for SHARE  based on its columns

    elsa_model = Models.Model_fixed_test_size(data=df, model_params=elsa_model_params, domain=domain_name, model='lgb', train_subset_size=1, order=0)
    evas = metric(elsa_model)
    evas_new = ELSA_additional_evals(elsa_model.y_test, elsa_model.test_set_predict_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas.imv, evas.auc_score, evas.pr_auc, evas.efron_rsquare, evas.pr_no_skill, evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]

    ELSA_test_set_id = df.loc[elsa_model.X_test.index, 'idauniq'].values

    del df, domain_name, evas, method, evas_new, ELSA_seed

    # block 2 ELSA+HRS
    method = 'ELSA+HRS'
    df, domain_lst = read_merged_data(type=3)
    domain_name = 'combine_all'
    model_params_comb = params.model_params
    model_params_comb['random_state'] = comb_seeds[seed_ind]
    domain_lst.sort()
    model_params_comb['domain_dict'][domain_name] = domain_lst
    combine_model = Models.Model_fixed_test_size(data=df, model_params=model_params_comb, domain=domain_name, model='lgb', train_subset_size=1, order=0)
    evas = metric(combine_model)
    evas_new = ELSA_additional_evals(combine_model.y_test, combine_model.test_set_predict_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas.imv, evas.auc_score, evas.pr_auc, evas.efron_rsquare, evas.pr_no_skill, evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]

    del method, evas, evas_new, df, domain_lst
    # block 3 ELSA part in the test set of ELSA+HRS

    method = 'ALL ELSA in ELSA+HRS test'
    df_comb, domain_lst = read_merged_data(type=3)
    domain_lst.sort()

    ELSA_ind_in_comb = df_comb.loc[df_comb['dataset'] == 2].index
    ELSA_ind_in_comb_test = combine_model.X_test.loc[combine_model.X_test.index.isin(ELSA_ind_in_comb)].index
    ELSA_in_comb_test_set_id = df_comb.loc[ELSA_ind_in_comb_test, 'idauniq'].values

    ELSA_pred = combine_model.model.predict(combine_model.X_test.loc[ELSA_ind_in_comb_test])
    ELSA_pred_prob = combine_model.model.predict_proba(combine_model.X_test.loc[ELSA_ind_in_comb_test])[:, 1]
    ELSA_true = combine_model.y_test.loc[combine_model.y_test.index.isin(ELSA_ind_in_comb)]
    evas = evaluate_metric(ELSA_true, ELSA_pred, ELSA_pred_prob, combine_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true, ELSA_pred_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]
    del evas, ELSA_pred, ELSA_pred_prob, ELSA_true, evas_new

    method = 'HRS in ELSA+HRS test'
    df_comb, domain_lst = read_merged_data(type=3)
    domain_lst.sort()

    HRS_ind_in_comb = df_comb.loc[df_comb['dataset'] == 0].index
    HRS_ind_in_comb_test = combine_model.X_test.loc[combine_model.X_test.index.isin(HRS_ind_in_comb)].index

    ELSA_pred = combine_model.model.predict(combine_model.X_test.loc[HRS_ind_in_comb_test])
    ELSA_pred_prob = combine_model.model.predict_proba(combine_model.X_test.loc[HRS_ind_in_comb_test])[:, 1]
    ELSA_true = combine_model.y_test.loc[combine_model.y_test.index.isin(HRS_ind_in_comb_test)]
    evas = evaluate_metric(ELSA_true, ELSA_pred, ELSA_pred_prob, combine_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true, ELSA_pred_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]
    del evas, ELSA_pred, ELSA_pred_prob, ELSA_true, evas_new

    # block 4 whole ELSA in ELSA+HRS
    method = 'All ELSA in ELSA+HRS all (including train set)'
    x = df_comb.loc[df_comb.index.isin(ELSA_ind_in_comb), domain_lst]
    y = df_comb.loc[df_comb.index.isin(ELSA_ind_in_comb), 'death']
    ELSA_pred = combine_model.model.predict(x)
    ELSA_pred_prob = combine_model.model.predict_proba(x)[:, 1]
    ELSA_true = y
    evas = evaluate_metric(ELSA_true, ELSA_pred, ELSA_pred_prob, combine_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true, ELSA_pred_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]

    del evas, ELSA_pred, ELSA_pred_prob, ELSA_true, evas_new

    # block 4 whole ELSA in ELSA+HRS
    method = 'All ELSA in ELSA (including train set)'
    df = params.data_reader(dataset='ELSA', source='us', bio=False)
    x = df[domain_lst]
    y = df['death']
    ELSA_pred = combine_model.model.predict(x)
    ELSA_pred_prob = combine_model.model.predict_proba(x)[:, 1]
    ELSA_true = y
    evas = evaluate_metric(ELSA_true, ELSA_pred, ELSA_pred_prob, combine_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true, ELSA_pred_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]

    # block 5 test set in ELSA prediction
    method = 'Test intersection in ELSA+HRS'
    df_ELSA = params.data_reader(dataset='ELSA', source='us', bio=False)  # df is ELSA+HRS, data is ELSA only
    ELSA_id_intersect = list(set(ELSA_in_comb_test_set_id).intersection(set(ELSA_test_set_id)))
    print(len(ELSA_id_intersect))

    x = df_comb.loc[df_comb['idauniq'].isin(ELSA_id_intersect), domain_lst]
    y = df_comb.loc[df_comb['idauniq'].isin(ELSA_id_intersect), 'death'].values
    ELSA_pred = combine_model.model.predict(x)
    ELSA_pred_prob = combine_model.model.predict_proba(x)[:, 1]
    ELSA_true = y

    evas = evaluate_metric(ELSA_true, ELSA_pred, ELSA_pred_prob, combine_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true, ELSA_pred_prob, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = [method, evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]



    # the same observations predicted by ELSA model
    x = df_ELSA.loc[df_ELSA['idauniq'].isin(ELSA_id_intersect), elsa_model_params['domain_dict']['elsa_all']]
    ELSA_pred_by_elsa_model = elsa_model.model.predict(x)
    ELSA_pred_prob_by_elsa_model = elsa_model.model.predict_proba(x)[:, 1]
    ELSA_true_by_elsa_model = df_ELSA.loc[df_ELSA['idauniq'].isin(ELSA_id_intersect), 'death'].values

    evas = evaluate_metric(ELSA_true, ELSA_pred_by_elsa_model, ELSA_pred_prob_by_elsa_model, elsa_model.y_train)
    evas_new = ELSA_additional_evals(ELSA_true_by_elsa_model, ELSA_pred_prob_by_elsa_model, threshold=threshold)
    df_elsa_in_comb.loc[len(df_elsa_in_comb),] = ['Test intersection in ELSA', evas['imv'], evas['roc_auc'], evas['auc'], evas['efron_r2'], evas['pr_no_skill'], evas_new['fpr'], evas_new['tpr'], evas_new['precision'], evas_new['lr']]

df_elsa_in_comb
temp = df_elsa_in_comb.groupby('method').mean().reset_index()
df_elsa_in_comb.to_csv(Path.cwd() / 'results/ELSA_model_comparisons_full.csv', index=False)
'''


import pandas as pd
import matplotlib.pyplot as plt
df_index = pd.read_stata("/Users/valler/Python/OX_Thesis/OX_thesis/data/ELSA/raw/index_file_wave_0-wave_5_v2.dta",convert_categoricals=False)
df = pd.read_csv("/Users/valler/Python/OX_Thesis/OX_thesis/data/ELSA/model_used_data/df_by_us_death_from_rand.csv")

df_index=df_index.loc[df_index['idauniq'].isin(df['idauniq']),]

df_index = df_index.merge(df[['death','rabyear','idauniq']], on='idauniq', how='left')
df['Zfatherseduc'].nunique()
