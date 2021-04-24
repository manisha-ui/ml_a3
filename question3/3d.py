import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X,y=load_digits(return_X_y=True)
pca=PCA(n_components=2)
pca.fit(X)

X = pca.transform(X)

a=X[:,0]
b=X[:,1]
print(a)
print(b)
fig, ax = plt.subplots()
for digit in np.unique(y):
    i=np.where(y==digit)
    ax.scatter(a[i],b[i],label=digit)

# print(X)
# print(X.shape)
# print(y.shape)
# plt.scatter(X,y)
ax.legend()
plt.savefig('q3.png')