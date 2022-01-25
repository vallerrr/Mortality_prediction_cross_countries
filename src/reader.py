import numpy as np
import pandas as pd
import os
from src import DataImport
import re
bio_08 = pd.read_stata(os.path.join(os.getcwd(),'biomarker', "BIOMK08BL_R.dta"))
bio_06 = pd.read_stata(os.path.join(os.getcwd(),'biomarker', "BIOMK06BL_R.dta"))

df = DataImport.data_reader()

df.reset_index(inplace=True)

# df_with_pid = pd.read_csv("/Users/valler/OneDrive - Nexus365/Dissertation/Data/hrsMort_drop_hhid.csv")

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




df['pn'] = 0
for index in df.index:
    row = re.findall('(\d{1,5})0(\d{2})',str(df['hhidpn'].iloc[index]))
    df.loc[index,'pn']=int(row[0][1])



test_06 = df.merge(bio_06,left_on=['hhidpn'],right_on=['hhidpn'])

test_08 = df.merge(bio_08,left_on=['hhidpn'],right_on=['hhidpn'])











'''
# Steps: 
# 1. delete samples that hhids are not in the df['hhid']

df_with_pid = pd.read_csv("/Users/valler/OneDrive - Nexus365/Dissertation/Data/hrsMort.cl20190131.csv", index_col=0)

df_hhid_lst = [int(x) for x in df['hhid']]

# df_with_pid_hhid_lst = ['int'(x.replace(' ','0')) for x in df_with_pid['hhid']]



i = 0
for index in df_with_pid.index:
    hhid = df_with_pid.loc[index,'hhid']
    hhid = int(hhid.replace(' ','0'))
    if hhid in df_hhid_lst:
        continue
    else:
        print('{} th row, drop row {}, hhid is {}'.format(i,index,hhid))
        df_with_pid.drop(index=index, inplace=True)
    i+= 1

df_with_pid.to_csv("/Users/valler/OneDrive - Nexus365/Dissertation/Data/hrsMort_drop_hhid.csv",index_col=0)

'''

'''

col_type = {'hhid': 'int' , 'pn': 'int' , 'cohort': 'int' ,
             'lb_qre2006': 'int' , 'lb_qre2008': 'int' , 'sampWeight': 'float', 'int_year': 'int' ,
             'int_month': 'int' , 'death_year': 'int' , 'death_month': 'int' , 'age': 'int' ,
             'agesq': 'int' , 'Zage': 'float', 'Zagesq': 'float', 'maleYN': 'int' , 'hispanicYN': 'int' ,
             'blackYN': 'int' , 'otherYN': 'int' , 'migrantYN': 'int' , 'Zfatherseduc': 'float',
             'Zmotherseduc': 'float', 'fathersocc': 'float', 'relocate': 'int' , 'finhelp': 'int' ,
             'fatherunemp': 'int' , 'repeatschool': 'int', 'physicalabuse': 'int' , 'substanceuse': 'int' ,
             'sumCAE': 'float', 'measles': 'int', 'mumps': 'int' , 'chickenpox': 'int' , 'asthma': 'int' ,
             'respdisorder': 'int' , 'speechimp': 'int' , 'allergic': 'int' , 'heartcond': 'int' ,
             'earcond': 'int' , 'migraines': 'int', 'stomachcond': 'int' , 'depression': 'int' ,
             'diabetes': 'int' , 'hypertension': 'int' , 'seizures': 'int' , 'Zwealth': 'float',
             'Zincome': 'float', 'rocc': 'float', 'everrent': 'int', 'evermedicaid': 'int' ,
             'everfoodstamp': 'int' , 'everunemployed': 'int' , 'everfoodinsec': 'int' ,
             'Zeduccat': 'float', 'Zrecentfindiff': 'float', 'Zneighsafety': 'float', 'Zneighcohesion': 'float', 
             'Zneighdisorder': 'float', 'vigactivityYN': 'int', 'modactivityYN': 'int' , 'alcoholYN': 'int' ,
             'sleepYN': 'int' , 'eversmokeYN': 'int' , 'currsmokeYN': 'int' , 'deathchild': 'int' ,
             'disaster': 'int' , 'combat': 'int' , 'partneraddict': 'int' , 'physattack': 'int' ,
             'spillness': 'int' , 'everwidowed': 'int' , 'sumadultAE': 'float', 'Zmajdiscrim': 'float', 
             'Zdailydiscrim': 'float', 'Znegchildren': 'float', 'Znegfamily': 'float', 'Znegfriends': 'float', 
             'Zposchildren': 'float', 'Zposfamily': 'float', 'Zposfriends': 'float', 'Zcompsocnetwork': 'float', 
             'everdivorced': 'int' , 'nevermarried': 'int' , 'Zagreeableness': 'float', 'Zangerin': 'float',
             'Zangerout': 'float', 'Zanxiety': 'float', 'Zconscientiousness': 'float', 'Zcynhostility': 'float', 
             'Zextroversion': 'float', 'Zhopelessness': 'float', 'Zlifesatis': 'float', 'Zloneliness': 'float', 
             'Znegaffect': 'float', 'Zneuroticism': 'float', 'Zopenness': 'float', 'Zoptimism': 'float', 
             'Zperceivedconstraints': 'float', 'Zperceivedmastery': 'float', 'Zpessimism': 'float', 'Zposaffect': 'float',
             'Zpurpose': 'float', 'Zreligiosity': 'float'}

for col in col_type.keys():
    if col_type[col] == 'float':
        df_with_pid[col] = pd.to_numeric(df_with_pid[col], errors='coerce')
        df_with_pid.dropna(subset=[col], inplace=True)
        df_with_pid[col] = df_with_pid[col].astype(float)
    elif col_type[col] == 'int':
        df_with_pid[col]=pd.to_numeric(df_with_pid[col], errors='coerce')
        df_with_pid.dropna(subset=[col], inplace=True)
        df_with_pid[col] = df_with_pid[col].astype(int)
'''


