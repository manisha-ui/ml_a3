
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import * 
from one_b import *
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing

(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


print("unregularised autograd")
for fit_intercept in [True, False]:
    LR = Logistic(fit_intercept=fit_intercept)
    LR.fit(X, y) 
    y_hat = LR.predict(X)
    print("accuracy", accuracy(y_hat,y))
