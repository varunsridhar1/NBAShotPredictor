import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#---------------------------------------------------
samples = pd.read_csv('./DUMMY_CODED.csv')

X_train, X_test, y_train, y_test = train_test_split(samples[['SHOT_NUMBER','GAME_CLOCK','SHOT_CLOCK','TOUCH_TIME','SHOT_DIST','CLOSE_DEF_DIST','FG%','DBPM','DAYS_SINCE_START','H','1','2','3','4','5','6']],samples['FGM'],test_size = 0.25, random_state=10)

def corr_class(ground_truth, predictions):
    mat=confusion_matrix(ground_truth,predictions)
    return (mat[0][0]+mat[1][1]*1.0)/np.sum(mat)

logReg2=LogisticRegression(penalty='l2')
cs=[0.01, 0.1, 1, 10 ,100]
tuned_parameters = [{'C': cs}]
n_folds = 3
scorerFunc=make_scorer(corr_class)

gridL = GridSearchCV(logReg2, tuned_parameters, cv=n_folds, refit=True, scoring = scorerFunc)
gridL.fit(X_train, y_train)

fpr, tpr, thresholds = roc_curve(gridL.predict(X_test), y_test)
auc = roc_auc_score(gridL.predict(X_test), y_test)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# This is the ROC curve
plt.show() 

print("Test True Positive Rate: ", corr_class(y_test, gridL.predict(X_test)))

#scoresGrid=gridL.cv_results_['mean_test_score']
#print("Average Per Class Accuracy for each C")
#print(scoresGrid)
#print("\nBest C value: "+ str(cs[np.argmax(scoresGrid)]) + ", Average Per Class Accuracy: "+ str(scoresGrid[np.argmax(scoresGrid)]))
