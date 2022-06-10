#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:03:10 2021

@author: valler
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
pd.options.mode.chained_assignment = None



def recode(var_dict, k):
    df = pd.DataFrame()
    values = {}
    i = 1
    for item in var_dict.items():
        # print(item[0],item[1][0],item[1][1])
        df1 = pd.read_stata(os.getcwd()+"/data_preprocess/Bio_data/" + item[0] + "/Stata/" + item[1][0] + ".dta")

        # respondent level
        if 'PN' in df1.columns:
            dashprint('this variable has PN')
            df1 = df1[['HHID', 'PN', item[1][1]]]
            # recode start------
            m = df1[item[1][1]].unique()

            for n in m:
                if ~np.isnan(n):
                    n = int(n)
                    if str(n) not in values.keys():
                        dashprint('\n This is year {}'.format(item[0]))
                        value = input('The value is {},please define the recode number of it\n'.format(n))
                        value = int(value)
                        values[str(n)] = value
                # replace
            for index, row in df1[[item[1][1]]].iterrows():
                if ~np.isnan(row[0]):
                    n = int(row[0])
                    if str(n) in values.keys():
                        # print(n)
                        value = values.get(str(n))

                        if value == k:
                            value = np.nan
                            df1.loc[index, item[1][1]] = value
                        df1.at[index, item[1][1]] = value
            if i == 1:
                df = df1
                # print(item[0])
            else:
                df = pd.merge(df, df1, how='outer', left_on=['HHID', 'PN'], right_on=['HHID', 'PN'])
            i = i + 1
        else:
            dashprint('this variable doesn\'t have PN')
            df1 = df1[['HHID', item[1][1]]]
            # recode start------
            m = df1[item[1][1]].unique()

            for n in m:
                if ~np.isnan(n):
                    n = int(n)
                    if str(n) not in values.keys():
                        dashprint('\n This is year {}'.format(item[0]))
                        value = input('The value is {},please define the recode number of it\n'.format(n))
                        value = int(value)
                        values[str(n)] = value
            # replace
            for index, row in df1[[item[1][1]]].iterrows():
                if ~np.isnan(row[0]):
                    n = int(row[0])
                    if str(n) in values.keys():
                        # print(n)
                        value = values.get(str(n))
                        if value == k:
                            value = np.nan
                            df1.loc[index, item[1][1]] = value
                        df1.at[index, item[1][1]] = value
            if i == 1:
                df = df1
                # print(item[0])
            else:
                df = pd.merge(df, df1, how='outer', left_on=['HHID'], right_on=['HHID'])
            i = i + 1
    return df


def dashprint(m):
    print('\n---------------------------------------')
    print(m)
    print('---------------------------------------\n')


def recodesave(transform, var_dict, df, j):
    # ---------------------------------------
    # save the df that has been recoded
    # j=0 : save both the original version and standardised version
    # ---------------------------------------

    transform = transform.replace(" ", "_").replace("/", "")

    if j == 0:
        # save the non-standardized version
        filename = os.getcwd()+"/data_preprocess/Bio_data/Variables/multi_original/" + transform + ".csv"
        df.to_csv(filename)
        print('non-standardized version saved')

        # save the standardized version
        transform = transform + '_stan'
        df = standardise(df)
        print('standardized version saved')
        print("{}=".format(transform))
        filename = os.getcwd()+"/data_preprocess/Bio_data/Variables/" + transform + ".csv"
        df.to_csv(filename)

    else:
        filename = os.getcwd()+"/data_preprocess/Bio_data/Variables/" + transform + ".csv"
        print("{}=".format(transform))
        df.to_csv(filename)


def standardise(df):
    if 'PN' in df.columns:
        m = 2  # Household Level
    else:
        m = 1  # Personal Level

    for column in df.columns[m:len(df.columns)]:
        df[column] -= np.mean(df[column], axis=0)
        df[column] /= np.std(df[column], axis=0)
        print('the column {} has been standardized and the mean is {:.3f}, std is {:.3f}'.format(column,np.mean(df[column],axis=0),np.std(df[column],axis=0)))
    return df


def multisave(df, varname, folder, PN):
    transform = varname.replace(" ", "_").replace("/", "")
    filename = os.getcwd()+"/data_preprocess/Bio_data/Variables/" + folder + transform + PN + ".csv"
    print(transform + PN)
    df.to_csv(filename)


def multirecode(H, year, reverse, reverse_dict, maxnan, obj, varname):
    # set the dictionary
    Dict = {}
    index = 1  # the number defined inside the variable

    for i in H:
        i = i.split("\t")
        Dict[index] = i[0], i[1], year
        index += 1

    # get variables and combine in df
    i = 1
    df = pd.DataFrame()
    for item in Dict.items():
        # item[1][0]: .dta file names
        # item[1][1]: variable names
        # item[1][2]: year
        df1 = pd.read_stata(os.getcwd()+"/data_preprocess/Bio_data/" + item[1][2] + "/Stata/" + item[1][0] + ".dta")

        if 'PN' in df1.columns:
            print('this is the {}th variable and it has PN'.format(i))
            PN = ""
            df1 = df1[['HHID', 'PN', item[1][1]]]
        else:
            print('this is the {}th variable and it doesn\'t have PN'.format(i))
            PN = "_H"
            df1 = df1[['HHID', item[1][1]]]

        if i == 1:
            df = df1
            df['zero_counts'] = 0
        else:
            df = pd.merge(df, df1, how='outer', left_on=['HHID', 'PN'], right_on=['HHID', 'PN'])
        # zero counts
        for index, row in df[[item[1][1]]].iterrows():
            if np.isnan(row[0]):
                df.loc[index, item[1][1]] = 0
                df.loc[index, 'zero_counts'] += 1
            else:
                if reverse[i - 1]:

                    n = int(row[0])
                    if n not in reverse_dict.keys():
                        print('\n This is year {}'.format(item[0]))
                        value = input('The value is {},please define the recode number of it\n'.format(n))
                        value = int(value)
                        reverse_dict[n] = value
                    else:
                        value = reverse_dict.get(row[0])

                    df.loc[index, item[1][1]] = value
        if reverse[i - 1]:
            print('\n This is year {} and this is the {}th variable, \nit is reversed coded'.format(item[1][2], i))
        i += 1

    # set teh rev mark
    if True in reverse:
        rev = ' rev'
    else:
        rev = ''

    # replace the column name
    key_list = list(Dict.keys())
    val_list = list(Dict.values())
    colname = {}

    for i in df.columns:
        for val in val_list:
            if i in val:
                colname[i] = 'var' + str(key_list[val_list.index(val)])

    df = df.rename(columns=colname)

    # Cronbach's Alpha
    N = len(df.filter(like='var').columns)
    df_corr = df.filter(like='var').corr()

    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr.loc[col][i + 1:].values
        rs = np.append(sum_, rs)

    mean_r = np.mean(rs)
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    dashprint("Cronbach's Alpha is {} \nand n is {} ".format(cronbach_alpha, N))

    # save to csv for the original data
    folder = 'multi_original/'
    # -----------------------------
    varname = varname + obj + rev
    # -----------------------------
    multisave(df, varname, folder, PN)

    # df=pd.read_csv("/Users/valler/codes/Replication/Bio_data/Variables/multi_original/22_Positive_and_Negative_Social_Interactions_Children_A_08_reversed.csv")

    # calculate the mean

    df1 = df[['HHID', 'PN']]
    df1['value'] = 0
    df.columns.str.startswith('v')
    for index, row in df.filter(like='var').iterrows():
        value = sum(row) / N
        if value == 0:
            df1.loc[index, 'value'] = np.nan
        else:
            df1.loc[index, 'value'] = value

    # standardise
    df1 = standardise(df1)
    varname += ' stan'
    # set missing
    # -----------------------------
    df1['value'] = np.where(df['zero_counts'] >= maxnan, np.nan, df1['value'])
    # -----------------------------
    # save to csv
    # multi_original
    folder = ''
    multisave(df1, varname, folder, PN)

def reverse_dict_generate(length):
    reverse_dict = {}
    for i in np.arange(1,length+1,1):
        reverse_dict[i]=length+1-i
    return reverse_dict

# ----------------------------------------------------------------------------
# This is the section for single variable recoding
# (draw information from only one vaiable)
# ----------------------------------------------------------------------------
'''

6_History_of_Food_Insecurity={
        #'1994':['W2D','W607'],
        '1996':['H96J_H','E4600'],
        '1998':['H98J_H','F5348'],
        '2000':['H00J_H','G5747'],
        '2002':['H02Q_H','HQ415'],
        '2004':['H04Q_H','JQ415'],
        '2006':['H06Q_H','KQ415'],
        '2008':['H08Q_H','LQ415']
        }


var_dict={'1996':['H96J_H','E4600'],
        '1998':['H98J_H','F5348'],
        '2000':['H00J_H','G5747'],
        '2002':['H02Q_H','HQ415'],
        '2004':['H04Q_H','JQ415'],
        '2006':['H06Q_H','KQ415'],
        '2008':['H08Q_H','LQ415']}
 
k = 0  # k is the number that should be replaced by nan
standardise_control = 1  # standardise_control=0: the variable should be standardized
df = recode(var_dict, k)


transform = '6_History_of_Food_Insecurity'

recodesave(transform, var_dict, df, standardise_control)

'''


# --------------------------------------------------------------------------------------------------
# below is the zone for variables recoded from multiple variables
# --------------------------------------------------------------------------------------------------
# specify zone
'''
varname = "46 Child Psychosocial Adversity "

H_08="H08LB_R	LLB037K\nH08LB_R	LLB037M\nH08LB_R	LLB037N"
H_06="H06LB_R	KLB037H\nH06LB_R	KLB037I\nH06LB_R	KLB037J"

item_num = [7,7]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": [True]*4,
                       "2008": [True]*4}
                       
# execute zone
var_dict = {"2006": [H_06, item_num[0]],
            "2008": [H_08, item_num[1]]}

for year in var_dict.keys():
    H = var_dict[year][0]
    reverse = reverse_dict_define[year]
    H = H.split("\n")
    item_num = var_dict[year][1]
    if not reverse_dict_define[year]:
        reverse = [False]*len(H)
    else:
        reverse = reverse_dict_define[year]
    reverse_dict = reverse_dict_generate(item_num)
    year_tail = year[2:4]

    multirecode(H, year, reverse, reverse_dict, maxnan, year_tail, varname)


'''

'''
---------------------------------------------------------------------------------------------------------------------
varname = "39 Daily Discrimination "
item_num = [6,6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": [True]*5,
                       "2008": [True]*6}

H_06 = "H06LB_R	KLB030A\nH06LB_R	KLB030B\nH06LB_R	KLB030C\nH06LB_R	KLB030D\nH06LB_R	KLB030E"
H_08 = "H08LB_R	LLB030A\nH08LB_R	LLB030B\nH08LB_R	LLB030C\nH08LB_R	LLB030D\nH08LB_R	LLB030E\nH08LB_R	LLB030F"
0.9472873262780296 
0.9546335314192461 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "38 Adult Psychosocial Adversity "
item_num = [2,2]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 6
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06 = "H06LB_R	KLB037A\nH06LB_R	KLB037B\nH06LB_R	KLB037C\nH06LB_R	KLB037D\nH06LB_R	KLB037E\nH06LB_R	KLB037G"
H_08 = "H08LB_R	LLB037A\nH08LB_R	LLB037B\nH08LB_R	LLB037C\nH08LB_R	LLB037D\nH08LB_R	LLB037E\nH08LB_R	LLB037G"
 0.9692064611729999
 0.9691880681933551 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "37 Trait Anxiety "
item_num = [4,4]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 3
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06 = "H06LB_R	KLB041A\nH06LB_R	KLB041B\nH06LB_R	KLB041C\nH06LB_R	KLB041D\nH06LB_R	KLB041E"
H_08 = "H08LB_R	LLB041A\nH08LB_R	LLB041B\nH08LB_R	LLB041C\nH08LB_R	LLB041D\nH08LB_R	LLB041E"
0.9625043527101937
0.9636041084767999 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "36 Perceived Mastery "
item_num = [6,6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06 = "H06LB_R	KLB023A\nH06LB_R	KLB023B\nH06LB_R	KLB023C\nH06LB_R	KLB023D\nH06LB_R	KLB023E"
H_08 = "H08LB_R	LLB023A\nH08LB_R	LLB023B\nH08LB_R	LLB023C\nH08LB_R	LLB023D\nH08LB_R	LLB023E"

0.9891426933197535 
0.9898938761011357
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "35 Perceived Constraints "
item_num = [6,6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": False,
                       "2008": False}
H_06 = "H06LB_R	KLB022A\nH06LB_R	KLB022B\nH06LB_R	KLB022C\nH06LB_R	KLB022D\nH06LB_R	KLB022E"
H_08 = "H08LB_R	LLB022A\nH08LB_R	LLB022B\nH08LB_R	LLB022C\nH08LB_R	LLB022D\nH08LB_R	LLB022E"

0.951375387925017 
0.9898938761011357
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "34 Religiosity "
item_num = [4,4]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 2
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06 = "H06LB_R	KLB028A\nH06LB_R	KLB028B\nH06LB_R	KLB028C\nH06LB_R	KLB028D"
H_08 = "H08LB_R	LLB028A\nH08LB_R	LLB028B\nH08LB_R	LLB028C\nH08LB_R	LLB028D"
0.9884539908552641 
0.9897721953334528 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "33 Purpose in Life "
item_num = [6, 6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 1
H_06 = "H06LB_R	KLB035A\nH06LB_R	KLB035B\nH06LB_R	KLB035C\nH06LB_R	KLB035D\nH06LB_R	KLB035E\nH06LB_R	KLB035F\nH06LB_R	KLB035G"
H_08 = "H08LB_R	LLB035A\nH08LB_R	LLB035B\nH08LB_R	LLB035C\nH08LB_R	LLB035D\nH08LB_R	LLB035E\nH08LB_R	LLB035F\nH08LB_R	LLB035G"
reverse_dict_define = {"2006": [False,True,False,True,True,True,False],
                       "2008": [False,True,False,True,True,True,False]}
0.9803371152624974
0.981919411998505
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
31 Positive Affect
0.9949654750053051
0.9980172920254722
H_06 = "H06LB_R	KLB027A\nH06LB_R	KLB027B\nH06LB_R	KLB027C\nH06LB_R	KLB027D\nH06LB_R	KLB027E\nH06LB_R	KLB027F"
maxnan = 3
H_08 = "H08LB_R	LLB027C\nH08LB_R	LLB027U\nH08LB_R	LLB027V\nH08LB_R	LLB027X\nH08LB_R	LLB027Y\nH08LB_R	LLB027D\nH08LB_R	LLB027F\nH08LB_R	LLB027G\nH08LB_R	LLB027H\nH08LB_R	LLB027K\nH08LB_R	LLB027P\nH08LB_R	LLB027Q\nH08LB_R	LLB027T"
maxnan = 7

reverse = [True]*len(H)
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
32 Negative Affect
0.9279750073052486 
0.9980107672397633
H_06 = "H06LB_R	KLB027I\nH06LB_R	KLB027J\nH06LB_R	KLB027K\nH06LB_R	KLB027L\nH06LB_R	KLB027M\nH06LB_R	KLB027N"
maxnan = 3
H_08 = "H08LB_R	LLB027A\nH08LB_R	LLB027R\nH08LB_R	LLB027S\nH08LB_R	LLB027W\nH08LB_R	LLB027B\nH08LB_R	LLB027E\nH08LB_R	LLB027I\nH08LB_R	LLB027J\nH08LB_R	LLB027L\nH08LB_R	LLB027M\nH08LB_R	LLB027N\nH08LB_R	LLB027O"
maxnan = 7

reverse = [True]*len(H)
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "41 Pessimism "
item_num = [6,6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 1
H_06 = "H06LB_R	KLB019F\nH06LB_R	KLB019J\nH06LB_R	KLB019K"
H_08 = "H08LB_R	LLB019F\nH08LB_R	LLB019J\nH08LB_R	LLB019K"

reverse_dict_define = {"2006": False,
                       "2008": False}
0.9279750073052486 
0.9226933125464439 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "30 Optimism "
item_num = [6,6]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 1
H_06 = "H06LB_R	KLB019G\nH06LB_R	KLB019H\nH06LB_R	KLB019I"
H_08 = "H08LB_R	LLB019G\nH08LB_R	LLB019H\nH08LB_R	LLB019I"

reverse_dict_define = {"2006": False,
                       "2008": False}
0.9751033134944616 
0.975355538645789            
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
29 Life Satisfaction

0.9827777767480689 
0.9793612158611602
H_06 = "H06LB_R	KLB003A\nH06LB_R	KLB003B\nH06LB_R	KLB003C\nH06LB_R	KLB003D\nH06LB_R	KLB003E"
H_08 = "H08LB_R	LLB003A\nH08LB_R	LLB003B\nH08LB_R	LLB003C\nH08LB_R	LLB003D\nH08LB_R	LLB003E"

maxnan = 3
reverse = [False]*len(H)
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
28 Loneliness 

06  0.9645252288290077 
H = "H06LB_R	KLB020A\nH06LB_R	KLB020B\nH06LB_R	KLB020C"
08 0.9591607190221784 
H="H08LB_R	LLB020A\nH08LB_R	LLB020B\nH08LB_R	LLB020C"


maxnan = 2
reverse = [True]*len(H)
reverse_dict = {1:3,2:2,3:1}
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
27 Hopelessness

06  0.94978110007212 
H = "H06LB_R	KLB019L\nH06LB_R	KLB019M\nH06LB_R	KLB019N\nH06LB_R	KLB019O"
08  0.9442024353575198 
H="H08LB_R	LLB019L\nH08LB_R	LLB019M\nH08LB_R	LLB019N\nH08LB_R	LLB019O"

maxnan = 3
reverse = [False]*len(H)
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
26 Cynical Hostility
06  0.9493448124111626 
H = "H06LB_R	KLB019D\nH06LB_R	KLB019E\nH06LB_R	KLB019A\nH06LB_R	KLB019B\nH06LB_R	KLB019C"

08 0.9468981342644204 
H="H08LB_R	LLB019A\nH08LB_R	LLB019B\nH08LB_R	LLB019C\nH08LB_R	LLB019D\nH08LB_R	LLB019E"

reverse = [False]*len(H)
maxnan = 4
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
25 Anger in
06 0.9663196010445981
H="H06LB_R	KLB042A\nH06LB_R	KLB042B\nH06LB_R	KLB042C\nH06LB_R	KLB042D"
08 0.9684051114161243 
H = "H08LB_R	LLB042A\nH08LB_R	LLB042B\nH08LB_R	LLB042C\nH08LB_R	LLB042D"


reverse = [False]*len(H)
maxnan = 3
varname = "25 Anger In "

---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
24 Anger out
06
H = "H06LB_R	KLB042E\nH06LB_R	KLB042F\nH06LB_R	KLB042G\nH06LB_R	KLB042H\nH06LB_R	KLB042I\nH06LB_R	KLB042J\nH06LB_R	KLB042K"
08
H = "H08LB_R	LLB042E\nH08LB_R	LLB042F\nH08LB_R	LLB042G\nH08LB_R	LLB042H\nH08LB_R	LLB042I\nH08LB_R	LLB042J\nH08LB_R	LLB042K"

reverse = [False]*len(H)
maxnan = 4
varname = "25 Anger  Out"


---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
23 Personal Traits

06
A="H06LB_R	KLB033F\nH06LB_R	KLB033G\nH06LB_R	KLB033H\nH06LB_R	KLB033I\nH06LB_R	KLB033J"
B="H06LB_R	KLB033A\nH06LB_R	KLB033B\nH06LB_R	KLB033C\nH06LB_R	KLB033D\nH06LB_R	KLB033E"
C="H06LB_R	KLB033P\nH06LB_R	KLB033Q\nH06LB_R	KLB033R\nH06LB_R	KLB033S"
D="H06LB_R	KLB033K\nH06LB_R	KLB033L\nH06LB_R	KLB033M\nH06LB_R	KLB033N\nH06LB_R	KLB033O"
E="H06LB_R	KLB033T\nH06LB_R	KLB033U\nH06LB_R	KLB033V\nH06LB_R	KLB033W\nH06LB_R	KLB033X\nH06LB_R	KLB033Y\nH06LB_R	KLB033Z"
08 
A="H08LB_R	LLB033B\nH08LB_R	LLB033F\nH08LB_R	LLB033J\nH08LB_R	LLB033O\nH08LB_R	LLB033V"
B="H08LB_R	LLB033A\nH08LB_R	LLB033E\nH08LB_R	LLB033I\nH08LB_R	LLB033S\nH08LB_R	LLB033W"
C="H08LB_R	LLB033C\nH08LB_R	LLB033G\nH08LB_R	LLB033K\nH08LB_R	LLB033P"
D="H08LB_R	LLB033D\nH08LB_R	LLB033H\nH08LB_R	LLB033M\nH08LB_R	LLB033T\nH08LB_R	LLB033Z"
E="H08LB_R	LLB033L\nH08LB_R	LLB033N\nH08LB_R	LLB033Q\nH08LB_R	LLB033R\nH08LB_R	LLB033U\nH08LB_R	LLB033X\nH08LB_R	LLB033Y"
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
22_Positive_and_Negative_Social_Interactions


Children 08 
A="H08LB_R	LLB008A\nH08LB_R	LLB008B\nH08LB_R	LLB008C"
B="H08LB_R	LLB008D\nH08LB_R	LLB008E\nH08LB_R	LLB008F\nH08LB_R	LLB008G"

Family 08 
A="H08LB_R	LLB012A\nH08LB_R	LLB012B\nH08LB_R	LLB012C"
B="H08LB_R	LLB012D\nH08LB_R	LLB012E\nH08LB_R	LLB012F\nH08LB_R	LLB012G"

Friends 08
A="H08LB_R	LLB016A\nH08LB_R	LLB016B\nH08LB_R	LLB016C"
B="H08LB_R	LLB016D\nH08LB_R	LLB016E\nH08LB_R	LLB016F\nH08LB_R	LLB016G"

Children 06
A="H06LB_R	KLB008A\nH06LB_R	KLB008B\nH06LB_R	KLB008C"
B="H06LB_R	KLB008D\nH06LB_R	KLB008E\nH06LB_R	KLB008F\nH06LB_R	KLB008G"

Family 06
A="H06LB_R	KLB012A\nH06LB_R	KLB012B\nH06LB_R	KLB012C"
B="H06LB_R	KLB012D\nH06LB_R	KLB012E\nH06LB_R	KLB012F\nH06LB_R	KLB012G"

Friends 06
A="H06LB_R	KLB016A\nH06LB_R	KLB016B\nH06LB_R	KLB016C"
B="H06LB_R	KLB016D\nH06LB_R	KLB016E\nH06LB_R	KLB016F\nH06LB_R	KLB016G"
---------------------------------------


---------------------------------------------------------------------------------------------------------------------
varname = "21 Neighborhood Cohesion "
item_num = [7,7]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06="H06LB_R	KLB021A\nH06LB_R	KLB021C\nH06LB_R	KLB021E\nH06LB_R	KLB021G"
H_08="H08LB_R	LLB021A\nH08LB_R	LLB021C\nH08LB_R	LLB021E\nH08LB_R	LLB021G"

0.9349100860795946 
0.9483944623711463 
---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------
varname = "42 Neighborhood Physical Disorder "

H_08="H08LB_R	LLB021B\nH08LB_R	LLB021D\nH08LB_R	LLB021F\nH08LB_R	LLB021H"
H_06="H06LB_R	KLB021B\nH06LB_R	KLB021D\nH06LB_R	KLB021F\nH06LB_R	KLB021H"
item_num = [7,7]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 4
reverse_dict_define = {"2006": [True]*4,
                       "2008": [True]*4}
0.8726448809416121 
0.9806103917487949 
---------------------------------------------------------------------------------------------------------------------

--------------------------------------
42_Lower_Father_Occupation
var_dict={
'2006':['H06B_R','KB024M'],
'2008':['H08B_R','LB024M']}

20_Neighborhood_Safety_20062008_H_stan=
{'2006': ['H06H_H', 'KH150'], 
 '2008': ['H08H_H', 'LH150']}

19_Recent_Financial_Difficulties_20062008_stan=
{'2006': ['H06LB_R', 'KLB039B'], 
 '2008': ['H08LB_R', 'LLB040']}

18_Education_1992_stan=
{'1992': ['HEALTH', 'V207']}


17_Paternal_and_Maternal_Education_Mother=
{'2002': ['H02B_R', 'HB027'],
 '2004': ['H04B_R', 'JB027'],
 '2006': ['H06B_R', 'KB027'],
 '2008': ['H08B_R', 'LB027']}


17_Paternal_and_Maternal_Education_Father_stan=
{'2002': ['H02B_R', 'HB026'],
 '2004': ['H04B_R', 'JB026'],
 '2006': ['H06B_R', 'KB026'],
 '2008': ['H08B_R', 'LB026']}


16_Sleep_Problems_20062008=
{'2006': ['H06D_R', 'KD112'], 
 '2008': ['H08RC_R', 'LRC112']}


15_Low_or_No_Vigorous_Activity_20062008=
{'2006': ['H06C_R', 'KC223'], 
 '2008': ['H08C_R', 'LC223']}


14_Low_or_No_Moderate_Activity=
{'2002': ['H02V_R', 'HV306'],
 '2004': ['H04C_R', 'JC224'],
 '2006': ['H06C_R', 'KC224'],
 '2008': ['H08C_R', 'LC224']}

13_History_of_Smoking_20062008={
'2006': ['H06C_R', 'KC116'], 
'2008': ['H08C_R', 'LC116']}

12_Current_Smoker={'2006': ['H06C_R', 'KC117'],
                   '2008': ['H08C_R', 'LC117']}

11_Alcohol_Abuse={
'2006':['H06C_R','KC135'],
'2008':['H08C_R','LC135']}

10_History_of_Medicaid={
        '1992':['HEALTH','V225'],
        '1994':['W2A','W200'],
        '1996':['H96CS_R','E375'],
        '1998':['H98A_R','F1071'],
        '2000':['H00A_R','G1158'],
        '2002':['H02B_R','HMARITAL'],
        '2004':['H04B_R','JB063'],
        '2006':['H06B_R','KB063'],
        '2008':['H08B_R','LB063']
        }

9_Never_Married={
        '1992':['HEALTH','V225'],
        '1994':['W2A','W200'],
        '1996':['H96CS_R','E375'],
        '1998':['H98A_R','F1071'],
        '2000':['H00A_R','G1158'],
        '2002':['H02B_R','HMARITAL'],
        '2004':['H04B_R','JB063'],
        '2006':['H06B_R','KB063'],
        '2008':['H08B_R','LB063']
        }


8_History_of_Divorce={
        '1992':['HEALTH','V225'],
        '1994':['W2A','W200'],
        '1996':['H96CS_R','E375'],
        '1998':['H98A_R','F1071'],
        '2000':['H00A_R','G1158'],
        '2002':['H02B_R','HMARITAL'],
        '2004':['H04B_R','JB063'],
        '2006':['H06B_R','KB063'],
        '2008':['H08B_R','LB063']
        }

10_History_of_Medicaid={
        #'1992':['HOUSEHLD','V6604'],
        #'1994':['W2R','W6700'],
        '1996':['H96R_R','E5136'],
        '1998':['H98R_R','F5869'],
        '2000':['H00R_R','G6242'],
        '2002':['H02N_R','HN006'],
        '2004':['H04N_R','JN006'],
        '2006':['H06N_R','KN006'],
        '2008':['H08N_R','LN006']
        }
7_History_of_Food_Stampsc={
        #'1994':['W2D','W607'],
        '1996':['H96J_H','E4588'],
        '1998':['H98J_H','F5348'],
        '2000':['H00J_H','G5735'],
        '2002':['H02Q_H','HQ400'],
        '2004':['H04Q_H','JQ400'],
        '2006':['H06Q_H','KQ400'],
        '2008':['H08Q_H','LQ400']
        }

6_History_of_Food_Insecurity={
        #'1994':['W2D','W607'],
        '1996':['H96J_H','E4600'],
        '1998':['H98J_H','F5348'],
        '2000':['H00J_H','G5747'],
        '2002':['H02Q_H','HQ415'],
        '2004':['H04Q_H','JQ415'],
        '2006':['H06Q_H','KQ415'],
        '2008':['H08Q_H','LQ415']
        }

5_History_of_Renting={
        #'1994':['W2D','W607'],
        '1996':['H96M_R','E5383'],
        '1998':['H98F_H','F2743'],
        '2000':['H00F_H','G3061'],
        '2002':['H02H_H','HH004'],
        '2004':['H04H_H','JH004'],
        '2006':['H06H_H','KH004'],
        '2008':['H08H_H','LH004']
        }

4_Relocated_Homes_in_Childhood={
        '1998':['H98A_R','F994'],
        '2000':['H00A_R','G1081'],
        '2002':['H02B_R','HB021'],
        '2004':['H04B_R','JB021'],
        '2006':['H06B_R','KB021'],
        '2008':['H08B_R','LB021']
        }

3_Family_Received_Financial_Help_in_Childhood={
        '1998':['H98A_R','F995'],
        '2000':['H00A_R','G1082'],
        '2002':['H02B_R','HB022'],
        '2004':['H04B_R','JB022'],
        '2006':['H06B_R','KB022'],
        '2008':['H08B_R','LB022']
        }


2_Father_was_Unemployed_in_Childhood={
        '1998':['H98A_R','F996'],
        '2000':['H00A_R','G1083'],
        '2002':['H02B_R','HB023'],
        '2004':['H04B_R','JB023'],
        '2006':['H06B_R','KB023'],
        '2008':['H08B_R','LB023']
        }

1_Black={
        '1992':['HEALTH','V221M'],
        '1994':['W2A','W233M'],
        '1996':['H96A_R','E667M'],
        '1998':['H98A_R','F1005A'],
        '2000':['H00A_R','G1092A'],
        '2002':['H02B_R','HB031A'],
        '2004':['H04B_R','JB031A'],
        '2006':['H06B_R','KB091M'],
        '2008':['H08B_R','LB091M']
        }


Hispanic={
        '1992':['HEALTH','V216'],
        '1994':['W2A','W228'],
        '1996':['H96A_R','E664'],
        '1998':['H98A_R','F1002A'],
        '2000':['H00A_R','G1089A'],
        '2002':['H02B_R','HB028A'],
        '2004':['H04B_R','JB028A'],
        '2006':['H06B_R','KB028'],
        '2008':['H08B_R','LB028']
        }
Sex={
        '1992':['HEALTH','V47'],
        '1994':['W2CS','W103'],
        '1996':['H96CS_R	','E374'],
        '1998':['H98CS_R	','F686'],
        '2000':['H00CS_R','G757'],
        '2002':['H02B_R','HB002'],
        '2004':['H04PR_R	','JX060_R'],
        '2006':['H06PR_R	','KX060_R'],
        '2008':['H08PR_R','LX060_R']
        }

        
USborn={
        '1992':['HEALTH', 'V204']
        '1994':['W2A', 'W215'],
        '1996':['H96A_R', 'E639'],
        '1998':['H98PR_R', 'F300'],
        '2000':['H00PR_R', 'G300'],
        '2002':['H02B_R', 'HB002'],
        '2004':['H04B_R', 'JB002'],
        '2006':['H06PR_R', 'KZ230'],
        '2008':['H08PR_R', 'LZ230']
        }
'''

