import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from metrics import *
from multiclass import *
from sklearn import preprocessing
from autograd import grad
(X,y)= load_digits(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print("gradient descent")
for fit_intercept in [True]:
    #print(fit_intercept)
    LR = Multiclass(fit_intercept=fit_intercept)
    LR.fit_logistic(X, y) 
    y_hat = LR.predict(X)
    #print(y_hat==y)
    print(fit_intercept)
    print("accuracy", accuracy(y_hat,y))

# print("autograd")
# for fit_intercept in [True]:
#     #print(fit_intercept)
#     LR = Multiclass(fit_intercept=fit_intercept)
#     LR.fit_autograd(X, y) 
#     y_hat = LR.predict(X)
#     #print(y_hat==y)
#     print(fit_intercept)
#     print("accuracy", accuracy(y_hat,y))

