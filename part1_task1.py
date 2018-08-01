# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:27:11 2018

@author: kazantseva
"""

import pandas as pd
import sklearn.linear_model as skl
import statsmodels.regression.linear_model as stm

data = pd.read_csv('mtcars.csv')

data = data.drop(data.columns[0], axis = 1)


model = skl.LinearRegression()
regr = model.fit(data.loc[:, 'cyl':], data.loc[:, 'mpg'])

model2 = stm.OLS(data.loc[:, 'mpg'], data.loc[:, 'cyl':])
regr2 = model2.fit()

res_sq = [res ** 2 for res in regr2.resid ]

model2 = stm.OLS(res_sq, data.loc[:, 'cyl':])
regr_res = model2.fit()