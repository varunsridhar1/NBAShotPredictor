# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:39:52 2017

@author: Varun
"""
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

print("XGBoost:")

samples = pd.read_csv('FINAL.csv')
features = ['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']

X_train, X_test, y_train, y_test = train_test_split(samples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

model0 = xgb.XGBClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model0, X_train, y_train, cv = kfold)
print("(all shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model0.fit(X_train, y_train)
importances = model0.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

def corr_class(ground_truth, predictions):
    mat=confusion_matrix(ground_truth,predictions)
    return (mat[0][0]+mat[1][1]*1.0)/np.sum(mat)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X_train.shape[1]))
plt.xlim([-1, X_train.shape[1]])
plt.show()

fpr, tpr, thresholds = roc_curve(model0.predict(X_test), y_test)
auc = roc_auc_score(model0.predict(X_test), y_test)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC')
plt.legend(loc="lower right")
# This is the ROC curve
plt.show() 

print("Test True Positive Rate: ", corr_class(y_test, model0.predict(X_test)))



twoptsamples = samples[samples['PTS_TYPE'] == 0]

X2_train, X2_test, y2_train, y2_test = train_test_split(twoptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],twoptsamples['FGM'],test_size = 0.25, random_state=10)

model1 = xgb.XGBClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model1, X2_train, y2_train, cv = kfold)
print("(2 pt shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model1.fit(X2_train, y2_train)
importances = model1.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X2_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X2_train.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X2_train.shape[1]), indices)
plt.xlim([-1, X2_train.shape[1]])
plt.show()

threeptsamples = samples[samples['PTS_TYPE'] == 1]

X3_train, X3_test, y3_train, y3_test = train_test_split(threeptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],threeptsamples['FGM'],test_size = 0.25, random_state=10)

model2 = xgb.XGBClassifier()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model2, X3_train, y3_train, cv = kfold)
print("(3 pt shots) Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model2.fit(X3_train, y3_train)
importances = model2.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X3_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X3_train.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X3_train.shape[1]), indices)
plt.xlim([-1, X3_train.shape[1]])
plt.show()

