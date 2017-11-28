# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:49:31 2017

@author: Varun
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

print("SVM:")

samples = pd.read_csv('FINAL.csv')
X_train, X_test, y_train, y_test = train_test_split(samples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],samples['FGM'],test_size = 0.95, random_state=10)

model0 = svm.SVC()
model0.fit(X_train, y_train)
predictY = model0.predict(X_test)
predictions = [round(value) for value in predictY]

accuracy = accuracy_score(y_test, predictions)
print("(all shots) Accuracy: %.2f%%" % (accuracy * 100.0))


twoptsamples = samples[samples['PTS_TYPE'] == 0]
X2_train, X2_test, y2_train, y2_test = train_test_split(twoptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],twoptsamples['FGM'],test_size = 0.95, random_state=10)

model1 = svm.SVC()
model1.fit(X2_train, y2_train)
predictY2 = model1.predict(X2_test)
predictions = [round(value) for value in predictY2]

accuracy = accuracy_score(y2_test, predictions)
print("(2 pt shots) Accuracy: %.2f%%" % (accuracy * 100.0))

threeptsamples = samples[samples['PTS_TYPE'] == 1]
X3_train, X3_test, y3_train, y3_test = train_test_split(threeptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],threeptsamples['FGM'],test_size = 0.95, random_state=10)

model2 = svm.SVC()
model2.fit(X3_train, y3_train)
predictY3 = model2.predict(X3_test)
predictions = [round(value) for value in predictY3]

accuracy = accuracy_score(y3_test, predictions)
print("(3 pt shots) Accuracy: %.2f%%" % (accuracy * 100.0))