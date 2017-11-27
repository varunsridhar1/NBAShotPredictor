# -*- coding: utf-8 -*-
"""
@author: Varun
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('FINAL3.csv')
df['HOME'] = df['LOCATION']
for i in range(len(df['LOCATION'])):
    if(df['LOCATION'][i] is 'H'):
        df['HOME'][i] = 1
    else:
        df['HOME'][i] = 0

df = df.drop('LOCATION', axis = 1)
df.to_csv('HomeAwayConverted.csv')