# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:39:52 2017

@author: Xan Frank
"""
import pandas as pd
import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

samples = pd.read_csv('./DUMMY_CODED.csv')
X = samples[['SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','H','1','2','3','4','5','6']]
y = samples['FGM']

X_train, X_test, y_train, y_test = train_test_split(samples[['SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','H','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

model1 = xgb.XGBClassifier(max_depth=3, n_estimators=17, learning_rate=0.05).fit(X_train, y_train)

predictY = model1.predict(X_test)
predictions = [round(value) for value in predictY]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

model2 = xgb.XGBClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model2, X, y, cv = kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
