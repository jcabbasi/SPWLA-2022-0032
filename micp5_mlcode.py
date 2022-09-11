# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:22:30 2021

@author: j
"""
import sklearn as svm
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
#import pyswarms as pso
#from pyswarms.utils.functions import single_obj as fx





dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
dfcpsize=dfcp.shape

idata=np.zeros((dfcpsize[0],5))
rdata=np.zeros((dfcpsize[0],1))


for coreindx in dfcp.index: 
   print(coreindx)
   popt=dfcp.loc[coreindx,'PCparameters']
   if popt=="ERROR":
       continue
   a,b,c=np.log10(popt)
   idata[coreindx,0]=dfcp.loc[coreindx,'Porosity'] 
   idata[coreindx,1]=dfcp.loc[coreindx,'Winland']                 
   idata[coreindx,[2,3,4]]=a,b,c
#   idata[coreindx,4]=dfcp.loc[coreindx,'Lithology']                   
   rdata[coreindx,0]=dfcp.loc[coreindx,'Permeability']


# Data Comparison:
x_train, x_test, y_train, y_test = train_test_split(idata, np.log10(rdata), random_state=1)

# Neural Network:
regrNN = MLPRegressor(random_state=1,solver="lbfgs", activation= "tanh", max_iter=5500).fit(x_train, y_train)
y_predNN=regrNN.predict(x_test)
scoreNN=regrNN.score(x_test, y_test)


# SVM Method:

#Optimization:
x0=[0.82]
def f1(c):
    xc=c
    xga=0.35
    xeps=0.11
    regr = svm.SVR(kernel='poly', C=xc, gamma=xga, epsilon=xeps)
    regr.fit(x_train, y_train)
    y_pred=regr.predict(x_test)
    scoreReg=1-regr.score(x_test, y_pred)
    return scoreReg

res = minimize(f1, x0,  method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)

regr = svm.SVR(kernel='poly', C=res.x, gamma=0.15, epsilon=.11)
regr.fit(x_train, y_train)
y_pred=regr.predict(x_test)
scoreReg=regr.score(x_test, y_pred)
scoreRegT=regr.score(x_train, y_train)


### PSO
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
def fpso(c):
    xc=c[0]
    xga=c[1]
    xeps=c[2]
    regr = svm.SVR(kernel='poly', C=xc, gamma=xga, epsilon=xeps)
    regr.fit(x_train, y_train)
    y_pred=regr.predict(x_test)
    scoreReg=1-regr.score(x_test, y_pred)
    return scoreReg


# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# Perform optimization
cost, pos = optimizer.optimize(fpso, iters=1000)



xideal=[-17,-13]
yideal= [-17,-13]

fig, ax2=plt.subplots(1,1)
ax2.scatter((y_test), (y_pred), color="red")
#ax2.scatter((y_test), (y_predNN), color="green")
ax2.plot((xideal), (yideal), color="black")
ax2.set_xlabel("K Measured"), ax2.set_ylabel("K Prediction")
#ax2.set_xscale('log'), ax2.set_yscale('log')

ax2.set_title("R2 SVM=" + str(scoreReg) + "  Train:" + str(scoreRegT) + ";  R2 NN=" +str(scoreNN))
#ax2.set_yscale('log')
#ax2.set_xscale('log')



