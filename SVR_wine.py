# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
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
    lens = len(b)
    return df ,lens
    
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
df = pd.read_csv(sep=';',filepath_or_buffer="winequality-white.csv",header=0,names=index_title)

print('確認是否有缺值：')
print(df.isnull().sum())#確認是否有缺值
print("--------------")
df , lens= checkvalue(df,index_title)
print(f'找到並刪除 {lens}')
print("--------------")

#%%兩個資料合併轉換#檢查相關性刪除低相關性
df = str_transform(df)
X,Y =  splt_X_Y(df)
X = np.array(X)
Y = np.array(Y)
X = X.reshape(4898,4)
Y = Y.reshape(4898,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.svm import SVR
from sklearn import metrics

clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, max_iter=500, shrinking=True, tol=0.001, verbose=True).fit(X_train, y_train)
p = clf.predict(X_test)
s = clf.score(X_test, y_test)
print("MAPE",np.mean(np.abs((y_test - p) / y_test)) * 100)
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test, p)))
