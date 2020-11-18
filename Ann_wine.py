# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:00:02 2020

@author: user
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split

from sklearn import metrics
#%%
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
def norm(df):
    df_norm = (df - df.mean()) / df.std()
    return  df_norm
        
def str_transform(df):
    #數值處理 最大駔小歸一化
    #字串處理 
    labelencoder = LabelEncoder()
    df['fixed_acidity'] = df['fixed_acidity']
    df['volatile_acidity'] =df['volatile_acidity']
    df['citric_acid'] = df['citric_acid']
    df['residual_sugar'] = df['residual_sugar']
    df['chlorides'] = df['chlorides']
    df['free_sulfur_dioxide'] = df['free_sulfur_dioxide']
    df['total_sulfur_dioxide'] = df['total_sulfur_dioxide']
    df['density'] = df['density']
    df['pH'] = df['pH']
    df['sulphates'] = df['sulphates']
    df['alcohol'] = df['alcohol']
    df['quality'] = df['quality']
    corrdf=df.corr()
    print(corrdf['quality'].sort_values(ascending=False))
    return df
def splt_X_Y(df):
    X = [df['alcohol'],
    df['pH'],
    df['sulphates'],
    df['free_sulfur_dioxide']]

    Y = [df['quality']]
    return X,Y
    
#%%
index_title=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"]
df = pd.read_csv(sep=';',filepath_or_buffer="winequality-white.csv",header=1,names=index_title)
#df2 = pd.read_csv(filepath_or_buffer="adult.train.txt",header=0,names=index_title)

print(df.isnull().sum())#確認是否有缺值
print("_____")
df , lens= checkvalue(df,index_title)
print(f'找到並刪除 {lens}')
#df2,lens2 = checkvalue(df2,index_title)
#print(f'找到並刪除 {lens2}')
print("_____")


#%%兩個資料合併轉換#檢查相關性刪除低相關性
df = str_transform(df)
X,Y =  splt_X_Y(df)
X = np.array(X)
Y = np.array(Y)
X=X.reshape(4897,4)
Y=Y.reshape(4897,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
'''
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
import tensorflow as tf

model = Sequential()
model.add(Dense(4,activation='relu',kernel_initializer='normal',input_dim=4))
model.add(Dense(2,activation='relu',kernel_initializer='normal'))
model.add(Dense(1,activation='linear',kernel_initializer='normal'))
model.summary()

#Adagrad = optimizers.Adagrad(lr=0.05, decay=0.0)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer= 'adam' , loss='mean_squared_error' , metrics=['mae'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs',profile_batch=1)
checkpoint = ModelCheckpoint('last.h5', verbose=2, monitor='loss',save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', mode = "min",patience=50,restore_best_weights=True)
model.fit(X_train,y_train,batch_size=32,validation_data=(X_test,y_test),epochs=3000,verbose=2,callbacks=[tensorboard_callback,checkpoint,early_stopping])
pred = model.predict(X_test)
s=model.predict_on_batch(X_test)
print("RMSE",np.mean(np.abs((y_test - pred) / y_test)) * 100)
print("MAPE",np.sqrt(metrics.mean_squared_error(y_test, pred)))
'''
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

clf = MLPClassifier(hidden_layer_sizes=16,activation="relu",early_stopping=True,random_state=32, max_iter=500,verbose=2).fit(X_train, y_train)
pa=clf.predict_proba(X_test[:1])
p=clf.predict(X_test)
s=clf.score(X_test, y_test)
print("RNSE",np.mean(np.abs((y_test - p) / y_test)) * 100)
print("MAPE",np.sqrt(metrics.mean_squared_error(y_test, p)))



