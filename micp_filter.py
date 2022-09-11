# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:52:49 2021

@author: j
"""

import xlrd 
import pandas as pd
import numpy as np

# Select only data from Prof Cai
dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
dfcpn=dfcp
#Only Select Sandstones
# dfcpn = dfcpn[dfcpn.Miner=='china' ]

#Only Select Sandstones
dfcpn = dfcpn[dfcpn.Lithology=='SS' ]

# Remove Outliers
# dfcpn = dfcpn[dfcpn.Outlier==False ]

dfcpn.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")



