import autograd.numpy as anp 
import numpy as np
import pandas as pd 
from autograd import grad

def sigmoid(z):
    return 1/(1+(np.e)**(-z))

def identity(z):
    return z

def relu(z):
    return np.maximum(0,z)


class nn():
    def __init__(self,neurons_list,acti_list):   
        self.neurons_list=neurons_list
        self.n_layers=len(neurons_list)
        self.acti_list=acti_list
    
    def init_weights(self):
        weights={}
        n_prevlayer=self.X.shape[1]
        for i in range(len(self.neurons_list)):
            a=np.ones((self.neurons_list[i],n_prevlayer+1)) 
            n_prevlayer=self.neurons_list[i]
            weights[i+1]=np.array(a)
        l=len(self.neurons_list)
        weights[len(self.neurons_list)+1]=np.array(np.ones((1,self.neurons_list[l-1]+1)))
        return weights

    def forward(self,weights):
        self.previous_layer_activation=(self.X)
        for i in range(len(weights)-1):  
            bias=np.ones(len(self.previous_layer_activation))
            self.previous_layer_activation=np.insert(self.previous_layer_activation,0,bias,axis=1)
            z=np.dot(self.previous_layer_activation,weights[i+1].T)    
            str=self.acti_list[i]
            if str=='relu':
                a=relu(z)
            elif str=='identity':
                a=identity(z)
            elif str=='sigmoid':
                a=sigmoid(z)            
            self.previous_layer_activation=a            
        bias=np.ones(len(self.previous_layer_activation))
        self.previous_layer_activation=np.insert(self.previous_layer_activation,0,bias,axis=1)

    def compute_error(self,weights):

        y_pred=np.array(sigmoid(np.dot(self.previous_layer_activation,weights[len(weights)].T)))
        return (np.square(np.subtract(self.y,y_pred)).mean())


    def fit(self,X,y,max_iters=300,lr=0.1):
        self.X=X
        self.y=y
        self.lr=lr
        weights=self.init_weights()
        for i in range(max_iters):
            self.forward(weights)
            grad_=grad(self.compute_error)
            c=grad_(weights)
            for j in range(len(weights)):
                weights[j+1]-=(self.lr)*(c[j+1])
        self.weights=weights




    def predict(self,X):
        previous_layer_activation= X
        for i in range(len(self.weights)-1):  
            bias=np.ones(len(previous_layer_activation))
            previous_layer_activation=np.insert(previous_layer_activation,0,bias,axis=1)
            z=np.dot(previous_layer_activation,self.weights[i+1].T)    
            str=self.acti_list[i]
            if str=='relu':
                previous_layer_activation=relu(z)
            elif str=='identity':
                previous_layer_activation=identity(z)
            elif str=='sigmoid':
                previous_layer_activation=sigmoid(z)                     
        bias=np.ones(len(previous_layer_activation))
        previous_layer_activation=np.insert(previous_layer_activation,0,bias,axis=1)
        y_hat=np.dot(previous_layer_activation,self.weights[len(self.weights)].T)
        return y_hat 
    



