import os
import pandas as pd
import numpy as np
from pathlib import Path
pd.options.mode.chained_assignment = None
from Variable_Acquire import multisave,standardise,dashprint


#0.9268990291166259
#0.9349409400538896
varname = "40 Major Discrimination "
item_num = [10,10]  # the point likert scale, used to generate the reverse dict (变量取值)
maxnan = 8
reverse_dict_define = {"2006": False,
                       "2008": False}

H_06 = "H06LB_R	KLB036A\nH06LB_R	KLB036B\nH06LB_R	KLB036C\nH06LB_R	KLB036D\nH06LB_R	KLB036E\nH06LB_R	KLB036F"
H_08 = "H08LB_R	LLB036A\nH08LB_R	LLB036B\nH08LB_R	LLB036C\nH08LB_R	LLB036D\nH08LB_R	LLB036E\nH08LB_R	LLB036F\nH08LB_R	LLB036G"
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

    year_tail = year[2:4]


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
            df1[item[1][1]].replace({1:1,5:-1},inplace=True)
            print(df1[item[1][1]].value_counts())
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

        i += 1


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
    varname = varname + year_tail
    # -----------------------------
    multisave(df, varname, folder, PN)


    # calculate the sum

    df1 = df[['HHID', 'PN']]
    df1['value'] = 0
    df.columns.str.startswith('v')
    for index, row in df.filter(like='var').iterrows():
        value = sum(row)
        if value == 0:
            df1.loc[index, 'value'] = 0
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





