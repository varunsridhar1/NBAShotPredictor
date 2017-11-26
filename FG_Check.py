import numpy as np
import pandas as pd
from scipy import stats, integrate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
data = pd.read_csv('./DUMMY_CODED.csv',sep=',')
print(data['DBPM'].isnull().sum())
data['DBPM'] = pd.to_numeric(data['DBPM'], errors='coerce')
arr = pd.isnull(data).any(1).nonzero()[0]
print(arr)
#for idx in arr:
#    data['DBPM'][idx] = -1.2
#data.to_csv('DUMMY_CODED.csv', index=False)
"""
fgTotal = 0
fgm = 0
for idx, player in enumerate(data['player_id']):
    if player == 202681:
        fgTotal = fgTotal + 1
        if data['FGM'][idx] == 1:
            fgm = fgm + 1
print(fgTotal)
print(fgm)
print(fgm / float(fgTotal))
"""