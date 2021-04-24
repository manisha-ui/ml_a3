# import autograd.numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn.datasets import load_breast_cancer
# from autograd import grad
# from metrics import *
# from LogisticRegresion import *


# (inputs,targets)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
# print("inputs")
# print(inputs)
# print("targets")
# print(targets)
# print(len(targets))
# targets_=[]
# for i in range(len(targets)):
#     print(targets[i]==1)
#     if(targets[i]==1):
#         targets_.append(1.0)
#     else:
#         targets_.append(0.0)
# print("targets")
# targets=np.array(targets_)
# print(targets)




# (inputs,targets)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
# targets_=[]
# for i in range(len(targets)):
#     print(targets[i]==1)
#     if(targets[i]==1):
#         targets_.append(1.0)
#     else:
#         targets_.append(0.0)
# targets=np.array(targets_)
# intercept = np.ones((inputs.shape[0], 1))
# inputs= np.hstack((intercept, inputs))
# weights =  np.zeros(inputs.shape[1])

# Build a toy dataset.

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)

X= np.array([[1,2,3,4],
              [1,2,3,4]])
y= np.array([0,1,2])
print(y.shape)
print(y.reshape(-1,1))
# Y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))
# print(Y_onehot)


#2A
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from autograd import grad
from metrics import *
from LogisticRegresion import *
from sklearn import preprocessing

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def L2_loss(weights):
    preds = logistic_predictions(weights, inputs)
    label_probabilities = (preds ** targets) * ((1 - preds) **(1 - targets))
    return -1*(np.sum(np.log(label_probabilities)))+ lamda*(np.dot(np.transpose(weights),weights))

def L1_loss(weights):
    preds = logistic_predictions(weights, inputs)
    label_probabilities = (preds ** targets) * ((1 - preds) **(1 - targets))
    return -1*(np.sum(np.log(label_probabilities))) + lamda* (np.absolute(np.sum(weights)))
 
# class L1L2():
#     def __init__(self,fit_intercept=True):
#         self.fit_intercept= fit_intercept
#     def L1_fit(X,y,n_iters=)

lamda=0.8
(inputs,targets)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
min_max_scaler = preprocessing.MinMaxScaler()
inputs = min_max_scaler.fit_transform(inputs)
targets_=[]
for i in range(len(targets)):
    if(targets[i]==1):
        targets_.append(1.0)
    else:
        targets_.append(0.0)
targets=np.array(targets_)
intercept = np.ones((inputs.shape[0], 1))
inputs= np.hstack((intercept, inputs))


# inputs = np.array([[1,2,3,4],
#                     [3.0,1,2.8,2],
#                     [1.2,5.6,7.1,1]])
# intercept = np.ones((inputs.shape[0], 1))
# inputs= np.hstack((intercept, inputs))
# targets = np.array([1,1,1])

weights_L1 =  np.ones(inputs.shape[1])
weights_L2= np.ones(inputs.shape[1])
print(weights_L2)
L2_training_gradient_fun = grad(L2_loss)
L1_training_gradient_fun = grad(L1_loss)
for i in range(1000):
    weights_L2 = weights_L2- L2_training_gradient_fun(weights_L2) * 0.01


res_L2 = sigmoid(np.dot(inputs,weights_L2))
for i in range(0,len(res_L2)):
    if(res_L2[i]>=0.5):
        res_L2[i]=1
    else:
        res_L2[i]=0


for i in range(1000):
    weights_L1 =weights_L1- L1_training_gradient_fun(weights_L1) * 0.01


res_L1 = sigmoid(np.dot(inputs,weights_L1))
for i in range(0,len(res_L1)):
    if(res_L1[i]>=0.5):
        res_L1[i]=1
    else:
        res_L1[i]=0



print("L1_accuracy")
print(accuracy(targets,np.array(res_L1)))
print("L2_accuracy")
print(accuracy(targets,np.array(res_L2)))

# print(weights_L1)
# print(weights_L2)
