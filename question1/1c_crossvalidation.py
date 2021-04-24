from sklearn.datasets import load_breast_cancer
from metrics import *
from LogisticRegresion import *
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.datasets import load_breast_cancer
from one_b import *
import pandas as pd
from sklearn import preprocessing

(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

k = 3
kf = KFold(n_splits=k, random_state=None,shuffle=True)
fit_acc=[]
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    LR = Logistic(fit_intercept=True)
    LR.fit(X_train, y_train) 
    y_hat = LR.predict(X_test)
    acc = accuracy(np.array(y_test),np.array(y_hat))
    fit_acc.append(acc)
print(fit_acc)
print(max(fit_acc))
print(sum(fit_acc)/k)
     
