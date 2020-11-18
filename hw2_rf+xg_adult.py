# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 02:27:58 2020

@author: USER
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xg
#刪問號
def checkvalue(df,index_title):
    a=[]
    b=[]
    for i in index_title:
        a.append(df[df[i]==' ?'].index.tolist())
    for i in a:
        if len(i)>0:
            for x in i:
                if x not in b:
                    b.append(x)
    df = df.drop(index=b)
    lens=len(b)
    return df ,lens

index_title=['age','workclass','fnlwgt','education','education_num','marital_status',
             'occupation','relationship','race','sex','capital_gain','capital_loss',
             'hours_per_week','native_country','alary']
df = pd.read_csv(filepath_or_buffer="adult.test.txt",header=0,names=index_title)
df2 = pd.read_csv(filepath_or_buffer="adult.train.txt",header=0,names=index_title)

print(df.isnull().sum())
print("------")
df,lens=checkvalue(df, index_title)
print(f'找到並刪除{lens}')
df2,lens2=checkvalue(df2, index_title)
print(f'找到並刪除{lens2}')

df = df.drop(columns=['native_country','alary'])
df2 = df2.drop(['native_country','alary'],axis=1)
data1 = pd.get_dummies(df)
data2 = pd.get_dummies(df2)

arr = np.array(data1,dtype = float)
np.random.shuffle(arr)
arr2 = np.array(data2,dtype = float)
np.random.shuffle(arr2)

X_train = np.delete(arr,5,1)
y_train = arr[:,5]
X_test = np.delete(arr2,5,1)
y_test = arr2[:,5]

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
                          n_estimators = 500, max_depth = 6, min_child_weight = 4, 
                          gamma = 0.5, subsample = 0.8, colsample_bytree = 0.7, 
                          reg_alpha = 3, reg_lambda = 3) 
XGModel.fit(X_train,y_train)

y_pred = XGModel.predict(X_test)
print('RMSE：' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
print('MAPE：' + str(MAPE(y_pred, y_test)))
