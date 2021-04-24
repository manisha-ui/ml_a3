
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from metrics import *
from multiclass import *
from sklearn import preprocessing
from autograd import grad
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
skf = StratifiedKFold(n_splits=4,random_state=None, shuffle=True)





import seaborn as sns


(X,y)= load_digits(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
a=[]

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR = Multiclass(fit_intercept=True)
    LR.fit_logistic(X_train, y_train,n_iters=60) 
    y_hat = LR.predict(X_test)
    acc= accuracy(y_hat,y_test)
    a.append(acc)
    print(confusion_matrix(y_test, y_hat))
    #print(y_hat==y)
    # 
    # 

print(a)
print(np.sum(a)/len(a))


svm=(sns.heatmap(confusion_matrix(y_test, y_hat), annot=True))
figure = svm.get_figure()    
figure.savefig('3c_heatmap.png', dpi=200)