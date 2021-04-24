import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as at # Thinly-wrapped numpy
from autograd import grad  
from autograd import elementwise_grad
from autograd.numpy import exp, log, sqrt


class LogisticRegression():
    def __init__(self, fit_intercept=True):
        
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass


    def sigmoid(self,z):
        return(1/(1+exp(-1*z)))
        

    def cost(self,theta): 
        number_of_samples= len(self.y)
        cost = -1*(1/number_of_samples)*( np.sum((np.dot(self.y,(log(self.sigmoid(np.dot(self.X,theta))))))) + np.sum((np.dot(1-self.y,(log(self.sigmoid(1-np.dot(self.X,theta))))))) )
        return cost
        


    def fit_logistic_unregularized(self, X, y, batch_size, n_iter=5000, lr=5e-5, lr_type='constant'):
        assert (len(X)==len(y))
        if self.fit_intercept :
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X_new = pd.merge(bias,X,left_index=True,right_index=True)      
        else:
            X_new = X.copy()
        self.X=X_new
        self.y=y
        #print(X_new)
        self.coef_ = np.zeros(X_new.shape[1])
        for i in range(1,n_iter+1):
            if (lr_type == 'constant'):
                alpha = lr
            else:
                alpha = lr/i
            per=np.random.permutation(len(X_new))
            X_new=X_new.iloc[per]
            y=y.iloc[per]
            X_batch=X_new[0:batch_size]
            y_batch=y[0:batch_size]
            self.coef_=self.coef_ - alpha*(np.dot(np.transpose(X_batch), self.sigmoid(np.dot(X_batch,self.coef_))-y_batch ))                
        pass
        

    def fit_logistic_autograd(self, X, y, batch_size, n_iter=300, lr=0.1, lr_type='constant'):
        assert (len(X)==len(y))
        if self.fit_intercept ==True:
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X_new = pd.merge(bias,X,left_index=True,right_index=True)      
        else:
            X_new = X.copy()
        self.X=X_new
        self.y=y

        self.coef_ = np.zeros(X_new.shape[1])
        for i in range(1,n_iter+1):
            if (lr_type == 'constant'):
                alpha = lr
            else:
                alpha = lr/i
            per=np.random.permutation(len(X_new))
            X_new=X_new.iloc[per]
            y=y.iloc[per]
            a=grad(self.cost)
            self.coef_= self.coef_  -  (alpha*a(self.coef_))
            
        pass

    def predict(self, X):
      X_new=X.copy()
      bias =pd.Series(np.ones(len(X_new)))
      if(self.fit_intercept==True):
          X_new.insert(loc=0,column='bias',value=bias)
      res = self.sigmoid(np.dot(X_new,self.coef_))
      for i in range(0,len(res)):
          if(res[i]>=0.5):
              res[i]=1
          else:
              res[i]=0
      return pd.Series(res) #y predicted = X.coef_

