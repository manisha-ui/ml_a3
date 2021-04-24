from one_b import *
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing

(X,y)= load_breast_cancer(return_X_y=True)   #sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X=X[:,:2]
 

 
# Initializing Classifiers
clf = Logistic(fit_intercept=True)

 
import matplotlib.pyplot as plt
import mlxtend
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
# %matplotlib inline  
 
gs = gridspec.GridSpec(3, 2)
 
fig = plt.figure(figsize=(14,10))
 
labels = ['Logistic Regression']
for clf, lab, grd in zip([clf],
                         labels,
                         [(0,0)]):
 
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
 
figure = fig.get_figure()    
figure.savefig('1d.png', dpi=200)
# plt.savfig('1d.jpg')