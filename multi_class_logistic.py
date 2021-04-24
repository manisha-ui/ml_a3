import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as at # Thinly-wrapped numpy
from autograd import grad  
from autograd import elementwise_grad
from metrics import *
# from scipy.special import softmax
# from scipy.special import log_softmax
from sklearn.preprocessing import OneHotEncoder
# from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)

def softmax(y):
    e_y = (np.e) ** (y - np.max(y))
    y=(e_y) / (e_y.sum()).reshape(-1,1)
    return y 

  
class multi_class_logistics():
    def __init__(self,fit_intercept=True):
        self.fit_intercept=fit_intercept


    def fit_autograd(self,X,y,n_iters=10,lr=0.1):
        self.y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))
        if(self.fit_intercept==True):
            bias= np.ones(len(X))
            X_new=np.insert(X,0,bias,axis=1)
        else:
            X_new=X
        self.X=X_new
        wts = np.zeros((self.X.shape[1], self.y_onehot.shape[1]))
        print(self.y_onehot.shape[1])
        print("I am here")
        # a=grad(self.loss)
        for i in range(n_iters):
            print("here")
            gradient=grad(self.a_loss)
            print("here also")
            b=gradient(wts)
            print("here again")


            wts -= lr*b
            print("here too")

        self.wts=wts

        pass
    def a_loss(self,wts):
        z=(np.dot(self.X,wts))
        pred=softmax(z)
        pred = np.log(pred)
        return np.mean(self.y_onehot*pred)


    def predict(self,X):
        if(self.fit_intercept==True):
            bias= np.ones(len(X))
            X_new=np.insert(X,0,bias,axis=1)
        else:
            X_new=X
        z = np.dot(X_new,self.wts)
        # print("pred")
        pred = softmax(z)
        # print("predicted")

        return (np.argmax(pred,axis=1))
    







        






