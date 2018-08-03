# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:08:13 2018

@author: kazantseva
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.regression.linear_model as smlr

def get_vif(exog):
  vif = {}
  for col in exog.columns:
    endog = exog.loc[:, col]
    exog_coef = exog.drop([col], axis = 1)
    exog_coef = sm.add_constant(exog_coef)
    model = smlr.OLS(endog, exog_coef).fit()
    vif[col] = 1 / (1 - model.rsquared)
  return vif

def smart_regression(df, ignore_cols):
  endog = df.iloc[:, 0]
  exog = df.iloc[:, 1:]
  exog = exog.drop(ignore_cols, axis = 1)  
  vifs = get_vif(exog)
  large_vifs = { k:vifs[k] for k in vifs.keys() if vifs[k] > 10 }
  coef = {}
  
  if len(large_vifs) != 0:   
    max_vif = max(large_vifs, key = large_vifs.get)
    ignore_cols.append(max_vif)
    coef = smart_regression(df, ignore_cols)
  else:
    exog = sm.add_constant(exog)
    model = smlr.OLS(endog, exog).fit()
    coef = model.params
  return coef

df = pd.read_csv('mtcars.csv')    
df = df.drop(df.columns[0], axis = 1)
coefs = smart_regression(df, [])
