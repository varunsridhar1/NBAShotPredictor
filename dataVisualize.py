import numpy as np
import pandas as pd
from scipy import stats, integrate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
data = pd.read_csv('./FINAL.csv',sep=',')

def get_sec(time_str):
    if (type(time_str) is float) or (type(time_str) is int):
        return time_str
    else: 
        m, s = time_str.split(':')
        return int(m)*60 + int(s) 
game_clock = []
#for idx,n in enumerate(data['GAME_CLOCK']):
#    game_clock.append(get_sec(data['GAME_CLOCK'][idx]))
touch_time = data['TOUCH_TIME']

shot_clock = data['SHOT_CLOCK'].dropna() 
dribbles = data['DRIBBLES']
shot_dist = data['SHOT_DIST']
close_def = data['CLOSE_DEF_DIST']
shot_num = data['SHOT_NUMBER']

zero_dribble = []
for idx, dribble in enumerate(dribbles):
    if dribble == 0 and touch_time[idx] >=0:
        zero_dribble.append(touch_time[idx])
zd_mean = np.mean(zero_dribble)
print(zd_mean)
print(len(touch_time[touch_time<0]))
"""
for idx,time in enumerate(touch_time):
    if time < 0:
        touch_time[idx] = zd_mean
data['TOUCH_TIME'] = touch_time.values
touch_time2 = data['TOUCH_TIME']
print(len(touch_time2[touch_time2<0]))
#data.to_csv('FINAL.csv', index=False)  

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
plt.figure()


d, w, t, g = train_test_split(data[['DRIBBLES']],touch_time,test_size = 0, random_state=20)
linReg = LinearRegression()
linReg.fit(d, t)
print('Correlation between Dribbles and Touch Time',linReg.score(d,t))

plt.boxplot(shot_dist)
sd_median = np.median(shot_dist)
sd_upper_quartile = np.percentile(shot_dist, 75)
sd_lower_quartile = np.percentile(shot_dist, 25)
sd_iqr = sd_upper_quartile - sd_lower_quartile
print("Shot Distance Upper Cutoff: ", (sd_upper_quartile+1.5*sd_iqr)) 
x = shot_dist[shot_dist>40]
print(np.max(x))
print("Shot Distance Lower Cutoff: ", (sd_lower_quartile-1.5*sd_iqr))
"""
#plt.show()


