import pandas as pd 
import numpy as np
data = pd.read_csv('./FINAL3.csv',sep=',')
#data.drop(['GAME_ID', 'MATCHUP', 'W', 'FINAL_MARGIN', 'DRIBBLES', 'PTS_TYPE', 'SHOT_RESULT', 'CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID', 'PTS', 'player_name', 'player_id'], axis = 1, inplace = True)
#data.to_csv('FINAL2.csv')
#shot_clock = data['SHOT_CLOCK']
#game_clock = data['GAME_CLOCK']
#
#def get_sec(time_str):
#    if (type(time_str) is float):
#        return time_str
#    else: 
#        m, s = time_str.split(':')
#        return int(m)*60 + int(s)        
#    
#no_shotclock_count = 0
#bad_count = 0
#
#print("data", data.shape)
#
#for idx,n in enumerate(shot_clock):
#    if np.isnan(n):
#        no_shotclock_count = no_shotclock_count + 1
#        if (get_sec(game_clock[idx]) > 24):
#            bad_count = bad_count + 1
#            data.drop(idx, inplace = True)
#            
#print("bad count", bad_count)
#print("total count", no_shotclock_count)
#print("data", data.shape)
Home = pd.get_dummies(data['LOCATION'])
Quarters = pd.get_dummies(data['PERIOD'])
data2 = pd.concat([data,Home,Quarters], axis=1)
data2.drop(['A',7,'PERIOD','LOCATION'], axis = 1, inplace = True)
data2.to_csv('DUMMY_CODED.csv')