# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:42:19 2017

@author: Varun
"""
from nba_py import player
import pandas as pd

df = pd.read_csv('shot_logs.csv')

players = df['player_name']
player_ids = df['player_id']

for i in range(20):
    print player_ids[i]

FG_PCT = []


#for i in range(len(players)):
 #   myPlayer = player.PlayerGeneralSplits(player_ids[i], season = '2014-15', per_mode = 'PerGame')
 #   FG_PCT.append(myPlayer.overall()['FG_PCT'])


