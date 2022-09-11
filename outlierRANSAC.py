# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:29:06 2021

@author: j
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")
dfcpo=dfcp[["Porosity","Permeability"]]


a = dfcpo.Porosity.to_numpy(dtype ='float32')
d = dfcpo.Permeability.to_numpy(dtype ='float32')/9.869233e-15
d=np.log10(d)

a=a.reshape(1, -1).T
d=d.reshape(1, -1).T


robust_estimator = RANSACRegressor(random_state=1)
robust_estimator.fit(a, d)
d_pred = robust_estimator.predict(a)

# calculate mse
mse = (d - d_pred) ** 2

index = np.argsort(mse.ravel())



fig, axes = plt.subplots(ncols=2)
axes[0].hist(mse,color='b')
axes[0].set_xlabel('RANSAC Score')
axes[0].set_ylabel('Frquency (log)')
axes[0].set_yscale('log')

colorcode=(mse.ravel()/1)
colorcodeshape=np.shape(colorcode)
colorc=np.ones((colorcodeshape[0],3))
colorc[:,2]=colorcode
outnum=272
anoind=index[-outnum:]
axes[1].scatter(a[index[:-outnum]], d[index[:-outnum]], c = colorcode[index[:-outnum]],s=300, label='inliers', alpha=0.75)
axes[1].scatter(a[index[-outnum:]], d[index[-outnum:]], c = colorcode[index[-outnum:]],s=300, label='outliers', alpha=0.75)
axes[1].set_xlabel('Porosity')
axes[1].set_ylabel('Permeability, mD (log)')

# axes[1].legend(loc=1)


anoind=np.append(anoind,[104]) #Observation of Fractal values
for coreindx in dfcp.index: 
   print(coreindx)
   if coreindx in anoind:
       dfcp.loc[coreindx,'Outlier']=bool(True)
   else:
       dfcp.loc[coreindx,'Outlier']=bool(False)


dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
