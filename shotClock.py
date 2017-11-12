import pandas as pd 
import numpy as np
data = pd.read_csv('./shot_logs.csv',sep=',')
shot_clock = data['SHOT_CLOCK']
game_clock = data['GAME_CLOCK']
def get_sec(time_str):
    if (type(time_str) is float):
        return time_str
    else: 
        m, s = time_str.split(':')
        return int(m)*60 + int(s)        
no_shotclock_count = 0
bad_count = 0
for idx,n in enumerate(shot_clock):
    if np.isnan(n):
        no_shotclock_count = no_shotclock_count + 1
        if (get_sec(game_clock[idx]) > 24):
            bad_count = bad_count + 1
            #print (idx,game_clock[idx])
        else:
            #print (game_clock[idx], shot_clock[idx])
            data['SHOT_CLOCK'][idx] = (get_sec(game_clock[idx]))
            #print (shot_clock[idx],get_sec(game_clock[idx]))
            #print (idx)
data.to_csv('updated_shotlogs.csv')
print (count)