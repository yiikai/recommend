import pandas as pd
import numpy as np
from numpy import *

head = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('ml-100k/u.data',sep='\t',names=head)
print data.head()
n_users = data.user_id.unique().shape[0]
n_items = data.item_id.unique().shape[0]
print 'users:%d,items:%d'%(n_users,n_items)

def map_r(x):
    if np.isnan(x):
        return 0
    else:
        return 1

def map_y(x):
    if np.isnan(x):
        return 0
    return x

movie_data = data.pivot(index='user_id',columns='item_id',values='rating')
y_data = movie_data.applymap(map_y)
r_data = movie_data.applymap(map_r)

# y = y_data, r =r_data
#use svd get features num
u,sigma,vt = linalg.svd(y_data)

#svd alg to get features (90% main ingredients)
def svd_features(sigma):
    sig2 = sigma**2
    total = sum(sig2)
    stand = total*0.9
    for i in range(len(sigma)):
        if sum(sig2[:i]) > stand:
            return i
    return None

nums = svd_features(sigma)
if nums == None:
    print 'features get error'
else:
    print "Fetures: %d" % nums

#X_data for features
x_data = pd.DataFrame(np.random.rand(n_users, nums))
#init theta for CF
theta = pd.DataFrame(np.random.rand(n_users,n_items))

def normalize_rate(y_data,r_data):
    nums = y_data.shape[0]
    for i in range(nums):
        idx = r_data[r_data.iloc[i,:] == 1].index
        total = 0
        for j in idx:
            total += y_data.iloc[i,j]
        meany = total/len(idx)
        print meany
        for j in idx:
            y_data.iloc[i,j] -= meany

normalize_rate(y_data,r_data)  #mean normalize

def DG(y_data,r_data,x_data,Theta,iters,lam):
    for i in range(iters):
        term = x_data * (Theta.T).dot(r_data) - y_data.dot(r_data)
        x_data = term * Theta +lam *x_data
        Theta = term.T*x_data+lam*Theta
        J = ((term.T)*term + lam*Theta.T*Theta + lam*x_data.T*x_data)/2
        print "lost: %f"%J

DG(y_data,r_data,x_data,theta,100,0.001)