# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:22:30 2021

@author: j
"""
from sklearn import svm
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn import preprocessing
#import pyswarms as pso
#from pyswarms.utils.functions import single_obj as fx
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import sklearn.metrics 
from averagePc import averagePc
from averagePc import rsaturation
from averagePc import satrange
from averagePc import modePc
from averagePc import satmin,medianPc
from scipy import stats


dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")
dfcpsize=dfcp.shape
WinlandPerm=dfcp.Winland.to_numpy(dtype='float64')/9.869233e-15
# dfcp.to_csv("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.csv")
dfcptno=np.reshape(dfcp.testNo.to_numpy(dtype='int'),[dfcp.shape[0],1])


irangec=np.linspace(0,20,0,dtype='int')
idatacNUM=len(irangec)

# idatacor=["Porosity","Winland","fractal","Purcell","Swanson"]
idatacor=["Porosity","Winland"]

dfcp.Winland=np.log10(dfcp.Winland.to_numpy(dtype='float64')/9.869233e-15)
dfcp.fractal=np.log10(dfcp.fractal.to_numpy(dtype='float64')/9.869233e-15)
dfcp.Swanson=np.log10(dfcp.Swanson.to_numpy(dtype='float64')/9.869233e-15)
dfcp.Purcell=np.log10(dfcp.Purcell.to_numpy(dtype='float64')/9.869233e-15)

srange=satrange(dfcp)
smin=satmin(dfcp)
medianPc=medianPc(dfcp)/100 #MPa
avgpcl=(averagePc(dfcp))/100
modePc=modePc(dfcp)


idata=np.zeros((dfcpsize[0],5))
rdata=np.zeros((dfcpsize[0],1))
idatacorlist=np.zeros([dfcp.shape[0],len(idatacor)],dtype='float64')
idatapclist=np.zeros([dfcp.shape[0],idatacNUM*1],dtype='float64')


# %% Cell A: Rank Radiuses

rdata= np.log10(dfcp.Permeability.to_numpy(dtype='float32')/9.869233e-15)
rsaturation20=np.log10(rsaturation(dfcp,0.41))
rsaturation80=np.log10(rsaturation(dfcp,0.5))
rsaturation50=np.log10(rsaturation(dfcp,0.6))
rsaturation65=np.log10(rsaturation(dfcp,0.7))
rsaturation75=np.log10(rsaturation(dfcp,0.8))
# rsaturation85=np.log10(rsaturation(dfcp,0.7)/9.869233e-15)
# rsaturation90=np.log10(rsaturation(dfcp,0.8)/9.869233e-15)
# rsaturation95=np.log10(rsaturation(dfcp,0.9)/9.869233e-15)
# rsaturation100=np.log10(rsaturation(dfcp,0.99)/9.869233e-15)

# rsaturation20=np.log10(rsaturation(dfcp,0.40)/9.869233e-15)
# rsaturation80=np.log10(rsaturation(dfcp,0.45)/9.869233e-15)
# rsaturation50=np.log10(rsaturation(dfcp,0.55)/9.869233e-15)
# rsaturation65=np.log10(rsaturation(dfcp,0.65)/9.869233e-15)
# rsaturation75=np.log10(rsaturation(dfcp,0.70)/9.869233e-15)


cornom=-1
for coreindx in dfcp.index: 
   cornom = cornom+1 
   popt=dfcp.loc[coreindx,'PCparameters']
   datapc=  ((popt[2,irangec].flatten()) )
   # datapc=  np.concatenate((popt[1,irangec].flatten() ,popt[2,irangec].flatten()) )
   idatapclist[cornom,:]=datapc
   idatacorlist[cornom,:]=dfcp.loc[coreindx,idatacor].to_numpy(dtype='float64')
   if popt=="ERROR":
       print(coreindx)
       continue
   # a,b,c=np.log10(popt)
   # idata[coreindx,0]=dfcp.loc[coreindx,'Porosity'] 
   # idata[coreindx,1]=dfcp.loc[coreindx,'Winland']                 
   # idata[coreindx,[2,3,4]]=a,b,c
   # rdata[coreindx,0]=dfcp.loc[coreindx,'Permeability']

idata=np.concatenate((idatapclist,idatacorlist),axis=1)
# idata=np.concatenate((idata,srange),axis=1)
# idata=np.concatenate((idata,smin),axis=1)
# idata=np.concatenate((idata,avgpcl),axis=1)
idata=np.concatenate((idata,medianPc),axis=1)
idata=np.concatenate((idata,rsaturation20),axis=1)
idata=np.concatenate((idata,rsaturation80),axis=1)
idata=np.concatenate((idata,rsaturation50),axis=1)
idata=np.concatenate((idata,rsaturation65),axis=1)
idata=np.concatenate((idata,rsaturation75),axis=1)
# idata=np.concatenate((idata,rsaturation85),axis=1)
# idata=np.concatenate((idata,rsaturation90),axis=1)
# idata=np.concatenate((idata,rsaturation95),axis=1)
# idata=np.concatenate((idata,rsaturation100),axis=1)




##
avgrsat=np.average(idata[:,range(3,8)],axis=1)


fig, axpc=plt.subplots(figsize=(7, 6.8))
axpc.set_yscale('log'), axpc.set_xscale('log')
axpc.scatter(np.power(10,rdata),np.power(10,avgrsat),color="blue",s=250,alpha=0.65)
axpc.set_xlabel("Log Measured Permeability (mD)"), axpc.set_ylabel("Log Average Radius (micrometer)")
tau, p_value = stats.kendalltau(rdata, avgrsat); print([tau,p_value])





# %% Cell A: Rank Radiuses

##

idata=np.concatenate((idata,dfcptno),axis=1)
# idata = preprocessing.scale(idata)
# Data Comparison:
x_train_s, x_test_s, y_train, y_test = train_test_split((idata), (rdata),test_size=0.2, random_state=7)

x_train=x_train_s[:,0:-1]
x_test=x_test_s[:,0:-1]
train_ind=x_train_s[:,-1]
test_ind=x_test_s[:,-1]
    

# Neural Network:
regrNN = MLPRegressor(hidden_layer_sizes=(20,22),random_state=5,solver="lbfgs",learning_rate="adaptive",alpha=0.0001,verbose=True, activation= "identity", max_iter=3000).fit(x_train, y_train)
y_predNN=regrNN.predict(x_test)
scoreNN=regrNN.score(x_test, y_test)
scoreNNTrain=regrNN.score(x_train, y_train)

print(scoreNN)
mseNNreg=sklearn.metrics.mean_absolute_percentage_error((y_predNN),(y_test))
print("MSE NN:"+ str(mseNNreg))




# PSO Algorithm:
maxbound=3*np.ones(3)
maxbound=[35,4,2]
minbound=np.zeros(3)
bounds=(minbound,maxbound)
# Initialize swarm
options = {'c1': 0.7, 'c2': 0.8, 'w':0.71}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options, bounds=bounds)

# SVM Method:

#Optimization:
x0=[40,5.0,0.06]

svmkernel="rbf"
def fpso(c,ga=10,ep=1):
    scoreReg=np.zeros(10)    
    for xx in  range(10):
        xc=c[xx,0]
        xga=c[xx,1]
        xeps=c[xx,2]
        regr = svm.SVR(kernel=svmkernel, C=xc, gamma=xga, epsilon=xeps)
        regr.fit(x_train, y_train)       
        scoreReg[xx]=1 -regr.score(x_test, y_test)
    
    return scoreReg


kwargs={"ga":x0[1],"ep":x0[2]}
cost, pos = optimizer.optimize(fpso, 600)

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()

regrPSO = svm.SVR(kernel=svmkernel, C=pos[0], gamma=pos[1], epsilon=pos[2])
regrPSO.fit(x_train, y_train)
ScoreTrainPSO=regrPSO.score(x_train, y_train)
y_trainPSO=regrPSO.predict(x_train)
scoreRegPSO=regrPSO.score(x_test, y_test)
scoreRegPSOTrain=regrPSO.score(x_train, y_train)

y_predPSO=regrPSO.predict(x_test)

sklearn.metrics.r2_score(np.power(10,y_predPSO),np.power(10,y_test))
sklearn.metrics.r2_score(np.power(10,y_predPSO),np.power(10,y_test))

msePSOSVM=sklearn.metrics.mean_absolute_percentage_error(y_predPSO,y_test)
msePSOSVMo=sklearn.metrics.mean_squared_error(y_predPSO,y_test)
msePSOSVMtr=sklearn.metrics.mean_squared_error(y_trainPSO,y_train)
tau, p_value = stats.kendalltau(y_train, y_trainPSO)



print("MSE SVM:"+ str(msePSOSVM))

def f1(c):
    xc=c[0]
    xga=1
    xeps=1
    regr = svm.SVR(kernel='rbf', C=xc, gamma=xga, epsilon=xeps)
    regr.fit(x_train, y_train)
    scoreReg=1 -regr.score(x_test, y_test)
    return scoreReg

res = minimize(f1, x0,  method='nelder-mead', options={'xatol': 1e-9, 'disp': True})
print(res.x)

regr = svm.SVR(kernel='rbf', C=res.x[0], gamma=res.x[1], epsilon=res.x[2])
regr.fit(x_train, y_train)
y_pred=regr.predict(x_test)
scoreRegT=regr.score(x_train, y_train)
scoreRegTest=regr.score(x_test, y_test)
print([scoreRegT,scoreRegTest])
mseSVMReg=sklearn.metrics.mean_absolute_percentage_error(y_pred,y_test)


fig, axt=plt.subplots()
axt.set_xscale('log')
axt.set_yscale('log')
axt.scatter(dfcp.Permeability,modePc/1000)



plt.rcParams.update({'font.size': 17})
plt.rcParams['font.family'] = 'arial'
fig, axall=plt.subplots(2,2)

# axall[0,0].scatter((y_test), (y_pred), color="red")
axall[0,0].set_yscale('log'), axall[0,0].set_xscale('log')
axall[0,0].scatter(np.power(10,y_test), np.power(10,y_predPSO), color="blue",s=200,alpha=0.65)
axall[0,0].scatter(np.power(10,y_train), np.power(10,y_trainPSO), color="orange",alpha=0.15)

# for i in range(len(test_ind)):
#     axall[0,0].text(y_test[i],y_predPSO[i] ,test_ind[i])
# axall[0,0].scatter((rdata), (np.log10(WinlandPerm)), color="black",alpha=0.2)
axall[0,0].plot(np.power(10,rdata),np.power(10,rdata),  ls="-")   
#ax2.scatter((y_test), (y_predNN), color="green")
# ax2.plot((xideal), (yideal), color="black")
axall[0,0].set_xlabel("Measured Permeability (mD)"), axall[0,0].set_ylabel("Predicted Permeability (mD)")
#ax2.set_xscale('log'), ax2.set_yscale('log')
m,b=np.polyfit(y_test,y_predPSO,1)
fity=m*y_test+b
axall[0,0].plot(np.power(10,y_test),np.power(10,fity),linewidth=5,  ls="-", alpha=1,color="red") 
trres="R-Square= {:.3} ".format(scoreRegPSO)+ "\n" + "Equation: y={:.2}x {:.2}".format(m,b)   
axall[0,0].text(np.min(np.power(10,y_test)),np.max(np.power(10,y_predPSO)) ,trres)


# axall[0,0].set_title("R2 SVM="  + str(scoreRegT) + ";  R2 SVM PSO:"+ str(scoreRegPSO) + ";  R2 NN=" +str(scoreNN))
#ax2.set_yscale('log')
#ax2.set_xscale('log')


# fig, ax4=plt.subplots(1,1)
axall[0,1].set_yscale('log')
axall[0,1].set_xscale('log')
axall[0,1].scatter(np.power(10,y_test), np.power(10,y_pred),s=200, color="red")
axall[0,1].scatter(np.power(10,y_test), np.power(10,y_predPSO),s=200, color="green")
axall[0,1].scatter(np.power(10,rdata), ((WinlandPerm)), color="black",s=100, alpha=0.2)
axall[0,1].plot(np.power(10,rdata),np.power(10,rdata), color="red",  ls="-")    

axall[0,1].set_xlabel("K Measured"), axall[0,1].set_ylabel("K Prediction")

axall[0,1].set_title("R2 SVM="  + str(scoreRegTest) + ";  R2 SVM PSO:"+ str(scoreRegPSO) + ";  R2 NN=" +str(scoreNN))

# fig, ax1=plt.subplots(1,1)
axall[1,0].scatter(y_test, y_predNN, s=250, color="blue",alpha=0.4,edgecolors="blue")
axall[1,0].plot(rdata,rdata,linewidth=3,  ls="--",alpha=1,color="orange")    
axall[1,0].set_xlabel("Measured Permeability (mD)"), axall[1,0].set_ylabel("Predicted Permeability (mD)")
m,b=np.polyfit(y_test,y_predNN,1)
fity=m*y_test+b
axall[1,0].plot(y_test,fity,linewidth=5,  ls="-", alpha=1,color="red") 
trres="R-Square= {:.3} ".format(scoreNN)+ "\n" + "Equation: y={:.2}x {:.2}".format(m,b)      
axall[1,0].text(np.min(y_test),np.max(fity) ,trres)



# fig, ax4=plt.subplots(1,1)
axall[1,1].set_yscale('log')
axall[1,1].set_xscale('log')
axall[1,1].scatter(np.power(10,y_train), np.power(10,y_trainPSO),s=250, color="blue",alpha=0.4,edgecolors="blue")
axall[1,1].plot(np.power(10,rdata),np.power(10,rdata),linewidth=3,  ls="--",alpha=1,color="blue")    
axall[1,1].set_xlabel("K Measured (mD)"), axall[1,1].set_ylabel("K Prediction (mD)")
m,b=np.polyfit(y_train,y_trainPSO,1)
fity=m*y_train+b
axall[1,1].plot(np.power(10,y_train),np.power(10,fity),linewidth=5,  ls="-", alpha=1,color="red")    
trres="R-Square= {:.3} ".format(scoreRegPSOTrain)+ "\n" + "Equation: y={:.2}x {:.2}".format(m,b)
axall[1,1].text(np.power(10,np.min(y_test)),np.power(10,np.max(y_predPSO) ),trres)



# Printing Results

print("Results /n")
print("Neural Network:")
print("Training Eff: "+ str(scoreNNTrain))
print("Test Eff:     "+ str(scoreNN))

print("----------")
print("SVM + PSO Opt:")
print("Training Eff: "+ str(scoreRegPSOTrain))
print("Test Eff:     "+ str(scoreRegPSO))
print("Test Eff MAPE:"+ str(msePSOSVM))
print("Test Eff MSE:"+ str(msePSOSVMo))

tau, p_value = stats.kendalltau(y_test, y_predPSO)
print("Kandal Tau:"+ str(tau))
print("P-Value:"+ str(p_value))


print("----------")
print("SVM + Regulat Opt:")
print("Training Eff: "+ str(scoreRegT))
print("Test Eff:     "+ str(scoreRegTest))
resexc=np.zeros([6,1])
resexc[0,0]=scoreRegPSOTrain
resexc[1,0]=scoreRegPSO
resexc[2,0]=msePSOSVM
resexc[3,0]=msePSOSVMo
resexc[4,0]=scoreNNTrain
resexc[5,0]=scoreNN



# %%

fig, ax4 = plt.subplots(nrows=1,ncols=2)
ax4[0].bar(['Phi-Perm','Winland',"PSO-SVM"],([0.77,0.98,0.076]), width = 0.8,color='blue',alpha=0.7)
ax4[0].set_xlabel("Algorithm"), ax4[0].set_ylabel("MSE (mD) ")
ax4[0].set_yscale('log')


lbls=["Porosirt","Winland","Swanson","Parachor","SVM-PSO"]




