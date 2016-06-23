# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:26:02 2016

@author: U505121
"""
import pandas as pd
from pandas import DataFrame
import nltk
import csv
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

df = pd.read_csv('C:/Users/U505121/Desktop/xml/10_K/final.csv')
df1=pd.concat([df['Indicator'], df['Phrase']], axis=1)


df_train = df[:1874]
df_test = df[1874:]

words = df_train['Phrase'].tolist()
words_test=df_test['Phrase'].tolist()
stopwords = ['a',
           'about',
           'above',
           'across',
           'after',
           'afterwards',
           'again',
           'against',
           'all',
           'almost',
           'alone',
           'along',
           'already',
           'also',
           'although',
           'always',
           'am',
           'among',
           'amongst',
           'amoungst',
           'amount',
           'an',
           'and',
           'another',
           'any',
           'anyhow',
           'anyone',
           'anything',
           'anyway',
           'anywhere',
           'are',
           'around',
           'as',
           'at',
           'back',
           'be',
           'became',
           'because',
           'become',
           'becomes',
           'becoming',
           'been',
           'before',
           'beforehand',
           'behind',
           'being',
           'below',
           'beside',
           'besides',
           'between',
           'beyond',
           'bill',
           'both',
           'bottom',
           'but',
           'by',
           'call',
           'can',
           'cannot',
           'cant',
           'co',
           'con',
           'could',
           'couldnt',
           'cry',
           'de',
           'describe',
           'detail',
           'do',
           'done',
           'down',
           'due',
           'during',
           'each',
           'eg',
           'eight',
           'either',
           'eleven',
           'else',
           'elsewhere',
           'empty',
           'enough',
           'etc',
           'even',
           'ever',
           'every',
           'everyone',
           'everything',
           'everywhere',
           'except',
           'few',
           'fifteen',
           'fify',
           'fill',
           'find',
           'fire',
           'first',
           'five',
           'for',
           'former',
           'formerly',
           'forty',
           'found',
           'four',
           'from',
           'front',
           'full',
           'further',
           'get',
           'give',
           'go',
           'had',
           'has',
           'hasnt',
           'have',
           'he',
           'hence',
           'her',
           'here',
           'hereafter',
           'hereby',
           'herein',
           'hereupon',
           'hers',
           'herself',
           'him',
           'himself',
           'his',
           'how',
           'however',
           'hundred',
           'i',
           'ie',
           'if',
           'in',
           'inc',
           'indeed',
           'interest',
           'into',
           'is',
           'it',
           'its',
           'itself',
           'keep',
           'last',
           'latter',
           'latterly',
           'least',
           'less',
           'ltd',
           'made',
           'many',
           'may',
           'me',
           'meanwhile',
           'might',
           'mill',
           'mine',
           'more',
           'moreover',
           'most',
           'mostly',
           'move',
           'much',
           'must',
           'my',
           'myself',
           'name',
           'namely',
           'neither',
           'never',
           'nevertheless',
           'next',
           'nine',
           'nobody',
           'none',
           'noone',
           'nor',
           'not',
           'nothing',
           'now',
           'nowhere',
           'of',
           'off',
           'often',
           'on',
           'once',
           'one',
           'only',
           'onto',
           'or',
           'other',
           'others',
           'otherwise',
           'our',
           'ours',
           'ourselves',
           'out',
           'over',
           'own',
           'part',
           'per',
           'perhaps',
           'please',
           'put',
           'rather',
           're',
           'same',
           'see',
           'seem',
           'seemed',
           'seeming',
           'seems',
           'serious',
           'several',
           'she',
           'should',
           'show',
           'side',
           'since',
           'sincere',
           'six',
           'sixty',
           'so',
           'some',
           'somehow',
           'someone',
           'something',
           'sometime',
           'sometimes',
           'somewhere',
           'still',
           'such',
           'system',
           'take',
           'ten',
           'than',
           'that',
           'the',
           'their',
           'them',
           'themselves',
           'then',
           'thence',
           'there',
           'thereafter',
           'thereby',
           'therefore',
           'therein',
           'thereupon',
           'these',
           'they',
           'thick',
           'thin',
           'third',
           'this',
           'those',
           'though',
           'three',
           'through',
           'throughout',
           'thru',
           'thus',
           'to',
           'together',
           'too',
           'top',
           'toward',
           'towards',
           'twelve',
           'twenty',
           'two',
           'un',
           'under',
           'until',
           'up',
           'upon',
           'us',
           'very',
           'via',
           'was',
           'we',
           'well',
           'were',
           'what',
           'whatever',
           'when',
           'whence',
           'whenever',
           'where',
           'whereafter',
           'whereas',
           'whereby',
           'wherein',
           'whereupon',
           'wherever',
           'whether',
           'which',
           'while',
           'whither',
           'who',
           'whoever',
           'whole',
           'whom',
           'whose',
           'why',
           'will',
           'with',
           'within',
           'without',
           'would',
           'yet',
           'you',
           'your',
           'yours',
           'yourself',
           'yourselves']
           
vect = CountVectorizer(stop_words = stopwords, token_pattern = '[a-z]+', min_df = 5, max_features = 100)
idfArray = vect.fit_transform(words).toarray()

vect_test = CountVectorizer(stop_words = stopwords, token_pattern = '[a-z]+', min_df = 5, max_features = 100)
testArray = vect_test.fit_transform(words_test).toarray()

number=preprocessing.LabelEncoder()
df['Phrase']=number.fit_transform(df.Phrase)
df['Indicator']=number.fit_transform(df.Indicator)
rf = RandomForestRegressor(n_estimators=100,oob_score=True)
rf.fit(idfArray, df['Indicator'][:1874])
t=[]
a=0
b=0
c=0
d=0
co=0
for i in range(0,len(testArray)):
    t.append(rf.predict(testArray))
for i in range(0,t.__len__()):
    t[i]=float(t[i])
df1=pd.DataFrame({'true':df['Indicator'][1873:],'model':t,'modify':1})  
for i in range(1873,2674):
    if df1['model'][i]>1.5:
        df1['modify'][i]=2
    else:
         df1['modify'][i]=1
    if df1['true'][i]==df1['modify'][i]:
        co=co+1
        
        
#creating confusion matrix
df2=pd.DataFrame(index={'Model +ve','Model -ve'},columns={'Target +ve','Target -ve'})
for i in range(1873,df1.__len__()+1873):
    if df1['modify'][i]==2 and df1['true'][i]==2:
        a=a+1
    elif df1['modify'][i]==2 and df1['true'][i]==1:
        b=b+1
    elif df1['modify'][i]==1 and df1['true'][i]==2:
        c=c+1
    else:
        d=d+1

df2['Target +ve']['Model +ve']=a
df2['Target +ve']['Model -ve']=c
df2['Target -ve']['Model +ve']=b
df2['Target -ve']['Model -ve']=d

pp=float(a)/(a+b)
np=float(c)/(c+d)
sen=float(a)/(a+c)
spe=float(b)/(b+d)
acc=(a+d)/float(a+b+c+d)