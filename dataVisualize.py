import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
data = pd.read_csv('./updated_shotlogs.csv',sep=',')

def get_sec(time_str):
    if (type(time_str) is float) or (type(time_str) is int):
        return time_str
    else: 
        m, s = time_str.split(':')
        return int(m)*60 + int(s) 
game_clock = []
for idx,n in enumerate(data['GAME_CLOCK']):
    game_clock.append(get_sec(data['GAME_CLOCK'][idx]))

touch_time = data['TOUCH_TIME']
shot_clock = data['SHOT_CLOCK'].dropna() 
dribbles = data['DRIBBLES']
shot_dist = data['SHOT_DIST']
close_def = data['CLOSE_DEF_DIST']
shot_num = data['SHOT_NUMBER']
sns.distplot(shot_clock)
sc_mean = mpatches.Patch(label= 'mean = %f' % shot_clock.mean())
sc_var = mpatches.Patch(label= 'variance = %f' % shot_clock.var())
plt.legend(handles=[sc_mean, sc_var])
plt.figure()

sns.distplot(touch_time, bins = 150)
tt_mean = mpatches.Patch(label= 'mean = %f' % touch_time.mean())
tt_var = mpatches.Patch(label= 'variance = %f' % touch_time.var())
plt.legend(handles=[tt_mean, tt_var])
plt.xlim(-1,24)
plt.figure()

sns.distplot(dribbles)
d_mean = mpatches.Patch(label= 'mean = %f' % dribbles.mean())
d_var = mpatches.Patch(label= 'variance = %f' % dribbles.var())
plt.legend(handles=[d_mean, d_var])
plt.xlim(0,30)
plt.figure()

sns.distplot(shot_dist)
sd_mean = mpatches.Patch(label= 'mean = %f' % shot_dist.mean())
sd_var = mpatches.Patch(label= 'variance = %f' % shot_dist.var())
plt.legend(handles=[sd_mean, sd_var])
plt.figure()

sns.distplot(close_def)
cd_mean = mpatches.Patch(label= 'mean = %f' % close_def.mean())
cd_var = mpatches.Patch(label= 'variance = %f' % close_def.var())
plt.legend(handles=[cd_mean, cd_var])
plt.figure()

sns.distplot(shot_num)
sn_mean = mpatches.Patch(label= 'mean = %f' % shot_num.mean())
sn_var = mpatches.Patch(label= 'variance = %f' % shot_num.var())
plt.legend(handles=[sn_mean, sn_var])
plt.figure()

sns.distplot(game_clock)
gc_mean = mpatches.Patch(label= 'mean = %f' % np.mean(game_clock))
gc_var = mpatches.Patch(label= 'variance = %f' % np.var(game_clock))
plt.legend(handles=[gc_mean, gc_var])
plt.show()