# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:39:22 2016

@author: U505121
"""
import pandas as pd
from pandas import DataFrame
import nltk
import csv
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
df = pd.read_csv('C:/Users/U505121/Desktop/xml/10_K/final.csv')
df1=pd.concat([df['Indicator'], df['Phrase']], axis=1)
t=[]
for i in range(0,df1.__len__()/2):
        t.append(word_tokenize(df1['Phrase'][i]))
f= [val for sublist in t for val in sublist]
for word in f: # iterate over word_list
  if word in stopwords.words('english'): 
    f.remove(word)
t=[]
for l in range(0,f.__len__()):
    if (re.search('^[^0-9]',f[l])):
        if (re.search('^[^-]',f[l])):
            if(re.search('^[^_]',f[l])):
                        t.append(f[l])
fdist = nltk.FreqDist(t)
k=[]
for word in fdist:
      if fdist [word]>30:
          k.append(word)
d={}
dicti=[]
for word in k:
    if re.search('[S|s]ale|[A|a]ccount|[R|r]evenue|[A|a]ttribute|[c|C]ontribute|[p|P]ercent|%',word):
        d={word:'True'}
    else:
         d={word:'False'}
    dicti.append(d)
result = {}
for d in dicti:
    result.update(d)