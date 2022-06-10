#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:52:55 2021

@author: valler
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import codecs
import urllib.request
with urllib.request.urlopen('http://python.org/') as response:
   html = response.read()


#1 
#read and get the do file comand, read and save the labels in the dta files
year=1992
result=[[]]

while year <= 2008:
    year1=str(year)
    mypath="/Users/valler/codes/Replication/Bio_data/"+year1+"/Stata"
    listdir(mypath) 
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    names=[f.replace('.DCT','') for f in onlyfiles if f.endswith('DCT') or f.endswith('dct')]
    names=[f.replace('.dct','') for f in names]
    names_inline=''
    for n in names:        
        names_inline += ' '+n
        
    text="foreach data in "+names_inline+" {"
    #print(names_inline)
    text1= "global path \""+mypath+"\""
    result.append([year1,[text+"\n"+text1]])
    
     
    #do file comands
    print(year1+"---------------------------------------")
    print(text)
    print(text1)
    print("\n")
    #'''
    #labels acquire
    label=pd.DataFrame(columns=["filename","key","explain"])
    for name in names:
        data=pd.read_stata("/Users/valler/codes/Replication/Bio_data/"+year1+"/Stata/"+name+".dta",iterator=True)
        dicti=data.variable_labels()
        for item in dicti.items():  
            label=label.append({"filename":name,"key":item[0],"explain":item[1]},ignore_index=True)
            label.to_csv("/Users/valler/codes/Replication/Bio_data/DataDescription/"+year1+".csv")
    #'''
    year+=2

'''
test=pd.read_stata("/Users/valler/codes/Replication/Bio_data/DTA/MODULEB.dta")
test1=pd.read_stata("/Users/valler/codes/Replication/Bio_data/DTA/MODULEA.dta")
test2=pd.read_stata("/Users/valler/codes/Replication/Bio_data/DTA/MODULEC.dta")

test3=pd.merge(test,test2,on='HHID',how='outer')
data=pd.merge(data,test1,on='HHID',how='outer')
'''


#read the txt files! 

def last2(n):
    return str(str(n)[-2:])


year=2006
result=[]
while year <= 2008:
    year1=str(year)
    mypath="/Users/valler/codes/Replication/Bio_data/"+year1+"/h"+str(last2(year))+"cb"
    listdir(mypath) 
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    i=1
    for file in onlyfiles:
        #buffer=open("/Users/valler/codes/Replication/Bio_data/"+year1+"/h"+str(last2(year))+"cb/"+file,'r')
        #txt=str = unicode(buffer.read(), errors='ignore')
        #result.append([txt])
        if i==1:
            df=pd.read_csv(mypath+"/"+file, sep='\t', encoding='ISO-8859-1',error_bad_lines=False,lineterminator='\n')
            df.columns=['1']
        df1=pd.read_csv(mypath+"/"+file, sep='\t',encoding='ISO-8859-1',error_bad_lines=False,lineterminator='\n')
        df1.columns=['1']
        df=df.append(df1,ignore_index=True)
        i+=1
    
    
    df.to_csv("/Users/valler/codes/Replication/Bio_data/"+year1+"/"+year1+".txt",sep='\t')
    year +=2



## read cross wave information
name = 'trk2018tr_r'
label=pd.DataFrame(columns=["filename","key","explain"])
data=pd.read_stata("/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Data/CrossWaveTracker/trk2018v2a/trk2018tr_r.dta",iterator=True)
dicti=data.variable_labels()
for item in dicti.items():
    label=label.append({"filename":name,"key":item[0],"explain":item[1]},ignore_index=True)
    label.to_csv("/Users/valler/Python/OX_Thesis/OX_thesis/data_preprocess/Bio_data/DataDescription/cross_wave.csv")
