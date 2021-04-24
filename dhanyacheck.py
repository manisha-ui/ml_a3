

import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn import preprocessing
from multiclass import *
from sklearn import preprocessing


def accuracy(y_hat,y):
    n=y.size
    equal=0
    for i in range(n):
        if y_hat[i]==y[i]:
            equal+=1
    return (1.0*equal)/n


(data,labels)=load_digits(return_X_y=True)
# scaler=preprocessing.MinMaxScaler()
# data=scaler.fit_transform(data)

n_class=len(np.unique(labels))
Obj=Multiclass(fit_intercept=True)
Obj.fit_autograd(data,labels)

y_hat=Obj.predict(data)
print(y_hat)
print(labels)
print("--------------------------------")
print(accuracy(y_hat,labels))


