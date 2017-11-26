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
samples = pd.read_csv('./DUMMY_CODED.csv')

X_train, X_test, y_train, y_test = train_test_split(samples[['SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','H','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

def func_auc(ground_truth, predictions):
    mat=confusion_matrix(ground_truth,predictions)
    return (mat[0][0]+mat[1][1]*1.0)/np.sum(mat)

logReg2=LogisticRegression(penalty='l2')
cs=[0.01, 0.1, 1, 10 ,100]
tuned_parameters = [{'C': cs}]
n_folds = 3
scorerFunc=make_scorer(func_auc)

gridL = GridSearchCV(logReg2, tuned_parameters, cv=n_folds, refit=False, scoring = scorerFunc)
gridL.fit(X_train, y_train)

scoresGrid=gridL.cv_results_['mean_test_score']
print("Average Per Class Accuracy for each C")
print(scoresGrid)
print("\nBest C value: "+ str(cs[np.argmax(scoresGrid)]) + ", Average Per Class Accuracy: "+ str(scoresGrid[np.argmax(scoresGrid)]))
