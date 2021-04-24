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

def predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

def loss(weights):
    preds = predictions(weights, inputs)
    label_probabilities = (preds * targets) + ((1 - preds) *(1 - targets))
    return -1*(np.sum(np.log(label_probabilities)))


# inputs = np.array([[1,2,3,4],
#                     [3.0,1,2.8,2]])

# intercept = np.ones((inputs.shape[0], 1))
# inputs= np.hstack((intercept, inputs))
# targets = np.array([0,1])
# weights =  np.zeros(inputs.shape[1])


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
weights =  np.zeros(inputs.shape[1])



gradient = grad(loss)
for i in range(1000):
    weights =weights-gradient(weights) * 0.01


res = sigmoid(np.dot(inputs,weights))
for i in range(0,len(res)):
    if(res[i]>=0.5):
        res[i]=1
    else:
        res[i]=0
print(res)
print(accuracy(targets,np.array(res)))

