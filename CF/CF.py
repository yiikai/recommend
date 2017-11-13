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

movie_data = data.pivot(index='item_id',columns='user_id',values='rating')
y_data = movie_data.applymap(map_y)
r_data = movie_data.applymap(map_r)

y = y_data.as_matrix()
r = r_data.as_matrix()

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
x_data = pd.DataFrame(np.random.rand(n_items, nums))
x = x_data.as_matrix()
#init theta for CF
theta = np.ones((n_users,nums))

print "x: ",np.shape(x)
print "y: ",np.shape(y)
print "r: ",np.shape(r)
print "theta: ",np.shape(theta)

def normalize_rate(y_data,r_data):
    nums = np.shape(y_data)[0]
    for i in range(nums):
        idx = np.where(r_data == 1)[1]
        total = 0
        for j in idx:
            total += y_data[i,j]
        meany = total/len(idx)
        for j in idx:
            y_data[i,j] -= meany

normalize_rate(y,r)  #mean normalize

def XTheta_DG(y_data,r_data,x_data,Theta,iters,lam,alpha):
    for i in range(iters):
        thetaT = np.transpose(Theta)
        term = np.dot(x_data,thetaT)*r_data - y_data*r_data
        #cost = (np.dot(x_data,thetaT)*r_data + lam*np.square(x_data) + lam*np.square(theta))/2
        grand_x = term * Theta + lam*x_data
        grand_theta = np.transpose(term)*x_data + lam*Theta
        x_data = x_data - alpha*grand_x
        Theta = Theta - alpha*grand_theta
    return Theta

result = XTheta_DG(y_data,r_data,x_data,theta,100,0.1,0.1)
print result