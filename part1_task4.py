# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:57:34 2018

@author: kazantseva
"""

import scipy.stats as sts
import pandas as pd
import numpy as np

norm_dist = sts.norm(10, 1)
x = np.array(norm_dist.rvs(10))
y = np.array(norm_dist.rvs(10))
corr_coef = {}

for p in np.arange(-2, 2, 0.1):  
  if p < 0:    
    x1 = -(x**p)
  elif p > 0:
    x1 = x**p
  else:
    x1 = np.log(x)
  corr = np.corrcoef(x1, y)[0][1]
  corr_coef[p] = corr

max_p = max(corr_coef, key = lambda p: np.abs(corr_coef[p]))  
print(max_p)
