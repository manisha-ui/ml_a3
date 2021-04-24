import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import * 
from LogisticRegresion import *
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
X=pd.DataFrame(X)
y=pd.Series(y)
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

# np.random.seed(42)
# N = 30
# P = 5
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))


print("Logistic unregularized")
for fit_intercept in [True, False]:
    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_logistic_unregularized(X, y,batch_size=100) 
    y_hat = LR.predict(X)
    print("fit intercept",fit_intercept)
    print("accuracy", accuracy(y_hat,y))


# # print ("***************************************************")

# print("autograd")
# for fit_intercept in [True, False]:
#     #print(fit_intercept)
#     LR = LogisticRegression(fit_intercept=fit_intercept)
#     LR.fit_logistic_autograd(X, y,batch_size=100) 
#     y_hat = LR.predict(X)
#     #print(y_hat)
#     #print(y)
#     #print(y_hat==y)
#     print(fit_intercept)
#     print("accuracy", accuracy(y_hat,y))

