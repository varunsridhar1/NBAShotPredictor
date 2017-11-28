import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#---------------------------------------------------
print("Logistic regression:")

samples = pd.read_csv('FINAL.csv')
features = ['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']

X_train, X_test, y_train, y_test = train_test_split(samples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

def corr_class(ground_truth, predictions):
    mat=confusion_matrix(ground_truth,predictions)
    return (mat[0][0]+mat[1][1]*1.0)/np.sum(mat)

logReg2=LogisticRegression(penalty='l2')
cs=[0.01, 0.1, 1, 10 ,100]
tuned_parameters = [{'C': cs}]
n_folds = 3
scorerFunc=make_scorer(corr_class)

gridL = GridSearchCV(logReg2, tuned_parameters, cv=n_folds, refit=False, scoring = scorerFunc)
gridL.fit(X_train, y_train)

scoresGrid=gridL.cv_results_['mean_test_score']
print("(all shots) Average Per Class Accuracy for each C")
print(scoresGrid)
print("\nBest C value: "+ str(cs[np.argmax(scoresGrid)]) + ", Average Per Class Accuracy: "+ str(scoresGrid[np.argmax(scoresGrid)]) + "\n")

twoptsamples = samples[samples['PTS_TYPE'] == 0]
X_train2, X_test2, y_train2, y_test2 = train_test_split(twoptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],twoptsamples['FGM'],test_size = 0.25, random_state=10)
gridL.fit(X_train2, y_train2)

scoresGrid=gridL.cv_results_['mean_test_score']
print("(2 pt shots) Average Per Class Accuracy for each C")
print(scoresGrid)
print("\nBest C value: "+ str(cs[np.argmax(scoresGrid)]) + ", Average Per Class Accuracy: "+ str(scoresGrid[np.argmax(scoresGrid)]) + "\n")

threeptsamples = samples[samples['PTS_TYPE'] == 1]
X_train3, X_test3, y_train3, y_test3 = train_test_split(threeptsamples[['LOCATION','SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','1','2','3','4','5','6']],threeptsamples['FGM'],test_size = 0.25, random_state=10)
gridL.fit(X_train3, y_train3)

scoresGrid=gridL.cv_results_['mean_test_score']
print("(3 pt shots) Average Per Class Accuracy for each C")
print(scoresGrid)
print("\nBest C value: "+ str(cs[np.argmax(scoresGrid)]) + ", Average Per Class Accuracy: "+ str(scoresGrid[np.argmax(scoresGrid)]) + "\n")