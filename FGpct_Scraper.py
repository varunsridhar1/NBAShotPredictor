# -*- coding: utf-8 -*-
"""
@author: Varun
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('FGpct.csv')

FG_PCT = {}

for i in range(len(df)):
    myPlayer = df['Player'][i].rsplit('\\', 1)[0]
    if myPlayer.lower() not in FG_PCT:
        FG_PCT[myPlayer.lower()] = df['FG%'][i]
    
df = pd.read_csv('updated_shotlogs.csv')
df['FG%'] = df['player_name']
for i in range(len(df)):
    if df['player_name'][i] in FG_PCT:
        df['FG%'][i] = FG_PCT[df['player_name'][i]]

df.to_csv('shot_logs_fgpct.csv')