# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 04:31:53 2020

@author: USER
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xg

df = pd.read_csv(filepath_or_buffer="winequality-white.csv",header=0,sep=';')
arr = np.array(df,dtype = float)
X = arr[:,:10]
Y = arr[:,11]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#loss
def MAPE(predict,target):
    return ( abs((target - predict) / target).mean()) * 100

#Random Forest
def train_and_predict_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    acc = model.score(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print('RMSE：' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    print('MAPE：' + str(MAPE(y_pred, y_test)))

model = RandomForestRegressor(max_depth = 9, n_estimators = 40, random_state=0)

train_and_predict_model(X_train, X_test, y_train, y_test, model)

#XGBoost
XGModel = xg.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.01, 
                          n_estimators = 500, max_depth = 7, min_child_weight = 4, 
                          gamma = 0.5, subsample = 0.8, colsample_bytree = 0.7, 
                          reg_alpha = 3, reg_lambda = 3) 
XGModel.fit(X_train,y_train)

y_pred = XGModel.predict(X_test)
print('RMSE：' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
print('MAPE：' + str(MAPE(y_pred, y_test)))
