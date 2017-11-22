# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:42:19 2017

@author: Varun
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('def_stats.csv')

DBPM = {}

for i in range(len(df)):
    myPlayer = df['Player'][i].rsplit('\\', 1)[0]
    if myPlayer.lower() not in DBPM:
        myPlayer = myPlayer.rsplit(' ', 1)[1] + ", " + myPlayer.rsplit(' ', 1)[0]
        DBPM[myPlayer] = df['DBPM'][i]
    
df = pd.read_csv('shot_logs_fgpct.csv')
df['DBPM'] = df['CLOSEST_DEFENDER']
for i in range(len(df)):
    if df['CLOSEST_DEFENDER'][i] in DBPM:
        df['DBPM'][i] = DBPM[df['CLOSEST_DEFENDER'][i]]

df.to_csv('shot_logs_fgpct_dbpm.csv')


