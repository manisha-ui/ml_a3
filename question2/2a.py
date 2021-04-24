import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import * 
from two import *
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing

(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)



print("L1 regularised")
for fit_intercept in [True,False]:
    LR = L1L2(fit_intercept=fit_intercept)
    LR.L1_fit(X, y) 
    y_hat = LR.predict(X)
    # print(y_hat)
    print(fit_intercept)
    print("accuracy", accuracy(y_hat,y))


print("L2 regularised")
for fit_intercept in [True, False]:
    LR = L1L2(fit_intercept=fit_intercept)
    LR.L2_fit(X, y) 
    y_hat = LR.predict(X)
    # print(y_hat)
    print(fit_intercept)
    print("accuracy", accuracy(y_hat,y))


