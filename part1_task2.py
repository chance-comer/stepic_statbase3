# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:41:04 2018

@author: kazantseva
"""

import pandas as pd
import statsmodels.regression.linear_model as stm

data = pd.read_csv('mtcars.csv')

data = data.drop(data.columns[0], axis = 1)

endog = data.loc[:, 'mpg']
exog = data.loc[:, 'cyl':]
const = [1] * len(exog)
exog['const'] = [1 for i in data.iterrows()]

model = stm.OLS(endog, exog)
modres = model.fit()

coefs = modres.params
params_vif = {}

for p in coefs.index:
  if p == 'const':
    continue
  coef_endog = data.loc[:, p]
  coef_exog = data.loc[:, [ ind for ind in coefs.index if ind not in [p, 'const'] ]]
  coef_exog['const'] = [1 for i in data.iterrows()]
  coef_model = stm.OLS(coef_endog, coef_exog).fit()
  params_vif[p] = 1 / (1 - coef_model.rsquared)