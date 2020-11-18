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
    df['age'] = df['age']
    df['workclass'] =labelencoder.fit_transform(df['workclass'])
    df['fnlwgt'] = df['fnlwgt']
    df['education'] = labelencoder.fit_transform(df['education'])
    df['education_num'] = df['education_num']
    df['marital_status'] = labelencoder.fit_transform(df['marital_status'])
    df['occupation'] = labelencoder.fit_transform(df['occupation'])
    df['relationship'] = labelencoder.fit_transform(df['relationship'])
    df['race'] = labelencoder.fit_transform(df['race'])
    df['sex'] = labelencoder.fit_transform(df['sex'])
    df['capital_gain'] = df['capital_gain']
    df['capital_loss'] = df['capital_loss']
    df['hours_per_week'] = df['hours_per_week']
    df['native_country'] = labelencoder.fit_transform(df['native_country'])
    df['salary'] = labelencoder.fit_transform(df['salary'])
    corrdf=df.corr()
    print(corrdf['hours_per_week'].sort_values(ascending=False))
    return df

def splt_X_Y(df):
    X = [df['age'],
    df['workclass'],
    df['education'],
    df['education_num'],
    df['sex'],
    df['capital_gain'],
    df['capital_loss'],
    df['salary']]

    Y = [df['hours_per_week']]
    return X,Y

#%%
index_title=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','salary']
df = pd.read_csv(filepath_or_buffer = "adult.test.txt", header = 0, names = index_title)
df2 = pd.read_csv(filepath_or_buffer = "adult.train.txt", header = 0, names = index_title)

print('確認是否有缺值：')
print(df.isnull().sum())#確認是否有缺值
print("--------------")
df , lens= checkvalue(df,index_title)
print(f'找到並刪除 {lens}')
df2,lens2 = checkvalue(df2,index_title)
print(f'找到並刪除 {lens2}')
print("--------------")

#%%兩個資料合併轉換#檢查相關性刪除低相關性
X,Y = splt_X_Y(str_transform(df.append(df2)))
X = np.array(X)
Y = np.array(Y)
X = X.reshape(45220,8)
Y = Y.reshape(45220,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.333, random_state=42)

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

clf = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=True, max_iter=500).fit(X_train, y_train)
p = clf.predict(X_test)
s = clf.score(X_test, y_test)
print("MAPE",np.mean(np.abs((y_test - p) / y_test)) * 100)
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test, p)))