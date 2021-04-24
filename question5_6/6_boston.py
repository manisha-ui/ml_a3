import numpy as np 
import pandas as pd 
from sklearn.datasets import load_boston 
from sklearn import preprocessing
from metrics import *
from five import *
from sklearn.model_selection import KFold 


def rmse(y_hat,y):
    return np.sqrt((np.square(np.subtract(y_hat,y)).mean()))


(X,y)=load_boston(return_X_y=True)
scaler=preprocessing.MinMaxScaler()
X=scaler.fit_transform(X)
y=np.matrix(y).T
nnet=nn([2,1],['relu','relu'])
nnet.fit(X,y)
y_hat=nnet.predict(X)
# print(y_hat)
print(rmse(y_hat,y))



#CROSS VALIDATION
(X,y)=load_boston(return_X_y=True)
scaler=preprocessing.MinMaxScaler()
X=scaler.fit_transform(X)
y=np.matrix(y).T
k = 3
kf = KFold(n_splits=k, random_state=None,shuffle=True)
rmse_list=[]
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    LR = nn([2,1],['relu','relu'])
    LR.fit(X_train, y_train) 
    y_hat = LR.predict(X_test)
    rms_err = rmse(np.array(y_test),np.array(y_hat))
    rmse_list.append(rms_err)
    # else:
    #   not_fic_acc.append(accuracy)
print(rmse_list)
print(min(rmse_list))
print(sum(rmse_list)/k)
