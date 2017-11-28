# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:18:16 2017

@author: Varun
"""

import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'

def get_sec(time_str):
    if (type(time_str) is float):
        return time_str
    else: 
        m, s = time_str.split(':')
        return int(m)*60 + int(s)    
    
df = pd.read_csv('BadClock.csv')
df['GAME_CLOCK'] = df['GAME_CLOCK'].apply(get_sec)

for i in range(len(df['SHOT_DIST'])):
    if float(df['SHOT_DIST'][i]) >= 23.75:
        df['PTS_TYPE'][i] = 1
    else:
        df['PTS_TYPE'][i] = 0

df_period = pd.get_dummies(df['PERIOD'])
df = pd.concat([df, df_period], axis = 1)
        
df.to_csv('FINAL.csv', index=False)
