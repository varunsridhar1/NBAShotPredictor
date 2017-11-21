# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:52:32 2017

@author: Sriram
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('FGpct.csv')

FG_PCT = {}

for i in range(len(df)):
    myPlayer = df['Player'][i].rsplit('\\', 1)[0]
    if myPlayer.lower() not in FG_PCT:
        FG_PCT[myPlayer.lower()] = df['FG%'][i]
    
df = pd.read_csv('shot_logs.csv')
df['FG%'] = df['player_name']
for i in range(len(df)):
    if df['player_name'][i] in FG_PCT:
        df['FG%'][i] = FG_PCT[df['player_name'][i]]

df.to_csv('shot_logs_fgpct.csv')