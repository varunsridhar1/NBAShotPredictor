# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:33:13 2017

@author: Varun
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print("GBDT:")

samples = pd.read_csv('FINAL.csv')

X_train, X_test, y_train, y_test = train_test_split(samples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

model0 = GradientBoostingClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model0, X_train, y_train, cv=10)
print("(all shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

twoptsamples = samples[samples['PTS_TYPE'] == 0]

X2_train, X2_test, y2_train, y2_test = train_test_split(twoptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],twoptsamples['FGM'],test_size = 0.25, random_state=10)

model1 = GradientBoostingClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model1, X2_train, y2_train, cv = kfold)
print("(2 pt shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

threeptsamples = samples[samples['PTS_TYPE'] == 1]

X3_train, X3_test, y3_train, y3_test = train_test_split(threeptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],threeptsamples['FGM'],test_size = 0.25, random_state=10)

model2 = GradientBoostingClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model2, X3_train, y3_train, cv = kfold)
print("(3 pt shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))