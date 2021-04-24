import autograd.numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.datasets import load_digits
from sklearn import preprocessing
from metrics import *
from autograd import grad





class Multiclass():

    def __init__(self,fit_intercept=True):
        self.fit_intercept=fit_intercept

    def loss(self,weights):
        Z = - self.X @ weights
        n=len(self.X)
        loss = (1/n)*(np.trace(self.X @ weights @ self.Y_onehot.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss


    def gradient(self,X, Y, weights):
        Z = - X @ weights
        P = softmax(Z, axis=1)
        gd = (X.T @ (Y - P)) 

        return gd

    
    def fit_logistic(self,X, Y, n_iters=100, lr=0.1):
        self.Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
        if(self.fit_intercept==True):
            bias= np.ones(len(X))
            X_new=np.insert(X,0,bias,axis=1)
        else:
            X_new=X
        self.X=X_new
        weights = np.zeros((self.X.shape[1], self.Y_onehot.shape[1]))
    
        for i in range(n_iters):
            weights -= lr * self.gradient(self.X, self.Y_onehot, weights)

        self.weights=weights
        return  weights

    

    def fit_autograd(self,X, Y, n_iters=1000, lr=0.1):
        self.Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
        if(self.fit_intercept==True):
            bias= np.ones(len(X))
            X_new=np.insert(X,0,bias,axis=1)
        else:
            X_new=X
        self.X=X_new
        weights = np.zeros((self.X.shape[1], self.Y_onehot.shape[1]))
    
        for i in range(n_iters):
            gradient_=grad(self.loss)
            weights -= lr * gradient_(weights)

        self.weights=weights
        return  weights

    def predict(self, X):
        if(self.fit_intercept==True):
            bias= np.ones(len(X))
            X_new=np.insert(X,0,bias,axis=1)
        else:
            X_new=X

        Z = - X_new @ self.weights
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
    

# (X,Y)= load_digits(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)


# fit model
# model = Multiclass()
# model.fit_logistic(X, Y)

# plot loss

# predict 
# y_hat=model.predict(X)

# # check the predicted value and the actual value
# print(accuracy(y_hat,Y))