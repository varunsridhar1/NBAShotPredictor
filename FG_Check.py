import numpy as np
import pandas as pd
from scipy import stats, integrate
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
data = pd.read_csv('./FINAL.csv',sep=',')
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