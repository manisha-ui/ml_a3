import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import * 
from two import *
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import KFold 


(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


k = 3
kf_out= KFold(n_splits=k, random_state=None,shuffle=True)
kf_in = KFold(n_splits=k, random_state=None,shuffle=True)
fit_acc=[]


l=[]
temp=0
for i in range(100):
    l.append(temp)
    temp+=0.01

accuracy_final=0
for train_index , test_index in kf_out.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    best=0
    validation=[]
    for lamda in l:
        acc=0
        for train_index_,test_index_ in kf_in.split(X_train):
            X_train_in ,X_val= X_train[train_index_],X_train[test_index_]
            y_train_in,y_val = y_train[train_index_],y_train[test_index_]
            LR = L1L2(fit_intercept=True)
            LR.L1_fit(X_train_in, y_train_in,lmda=lamda) 
            y_hat = LR.predict(X_val)
            acc += accuracy((y_val),(y_hat))
        if(acc>best):
            # print("yes")
            best = acc
            best_parameter= lamda
        validation.append(acc/3)
    LR.L1_fit(X_train, y_train,lmda=best_parameter) 
    y_hat = LR.predict(X_test)
    accuracy_final += accuracy((y_hat),(y_test))

print(best_parameter)
print(accuracy_final/3)


    
        
    


        


        
            












# for train_index , test_index in kf.split(X):
#     X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
#     y_train , y_test = y[train_index] , y[test_index]
#     LR = LogisticRegression(fit_intercept=True)
#     LR.fit_logistic_unregularized(X_train, y_train,batch_size=200) 
#     y_hat = LR.predict(X_test)
#     acc = accuracy(np.array(y_test),np.array(y_hat))

   
#     fit_acc.append(acc)
#     # else:
#     #   not_fic_acc.append(accuracy)
# print(fit_acc)
# print(max(fit_acc))
# print(sum(fit_acc)/k)
     

