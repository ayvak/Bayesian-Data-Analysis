# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:26:02 2016

@author: U505121
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
boston = load_boston()
df['Phrase']=number.fit_transform(df.Phrase)
df['Indicator']=number.fit_transform(df.Indicator)
rf = RandomForestRegressor(n_estimators=20,oob_score=True)
rf.fit(df['Phrase'][:1872,None], df['Indicator'][:1872])
t=[]
c=0
for i in range(1873,df.__len__()):
    t.append(rf.predict(df['Phrase'][i]))
for i in range(0,t.__len__()):
    t[i]=float(t[i])
df1=pd.DataFrame({'true':df['Indicator'][1873:],'model':t,'modify':1})  
for i in range(1873,2674):
    if df1['model'][i]>1.5:
        df1['modify'][i]=2
    else:
         df1['modify'][i]=1
    if df1['true'][i]==df1['modify'][i]:
        c=c+1