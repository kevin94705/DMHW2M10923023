import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split
import keras
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
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return  df_norm
        
def str_transform(df):
    #數值處理 最大駔小歸一化
    #字串處理 
    labelencoder = LabelEncoder()
    df['age'] = norm(df['age'])
    df['workclass'] = norm(labelencoder.fit_transform(df['workclass']))
    df['fnlwgt'] = norm(df['fnlwgt'])
    df['education'] = norm(labelencoder.fit_transform(df['education']))
    df['education_num'] = norm(df['education_num'])
    df['marital_status'] = norm(labelencoder.fit_transform(df['marital_status']))
    df['occupation'] = norm(labelencoder.fit_transform(df['occupation']))
    df['relationship'] = norm(labelencoder.fit_transform(df['relationship']))
    df['race'] = norm(labelencoder.fit_transform(df['race']))
    df['sex'] = labelencoder.fit_transform(df['sex'])
    df['capital_gain'] = norm(df['capital_gain'])
    df['capital_loss'] = norm(df['capital_loss'])
    df['hours_per_week'] = df['hours_per_week']
    df['native_country'] = norm(labelencoder.fit_transform(df['native_country']))
    df['salary'] = labelencoder.fit_transform(df['salary'])
    return splt_X_Y(df)
def splt_X_Y(df):
    X = [df['age'],
    df['workclass'],
    df['fnlwgt'],
    df['education'],
    df['education_num'],
    df['marital_status'],
    df['occupation'],
    df['relationship'],
    df['race'],
    df['sex'],
    df['capital_gain'],
    df['capital_loss'],
    df['native_country'],
    df['salary']]

    Y = [df['hours_per_week']]
    return X,Y
    
#%%
index_title=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','salary']
df = pd.read_csv(filepath_or_buffer="adult.test.txt",header=0,names=index_title)
df2 = pd.read_csv(filepath_or_buffer="adult.train.txt",header=0,names=index_title)

print(df.isnull().sum())#確認是否有缺值
print("_____")
df , lens= checkvalue(df,index_title)
print(f'找到並刪除 {lens}')
df2,lens2 = checkvalue(df2,index_title)
print(f'找到並刪除 {lens2}')
print("_____")
#%%兩個資料合併轉換
X,Y= str_transform(df.append(df2))
X = np.array(X)
Y = np.array(Y)
X=X.reshape(45220,14)
Y=Y.reshape(45220,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)

#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import optimizers
import tensorflow as tf
model = Sequential()
model.add(Dense(units=64,
                  activation='relu',
                  kernel_initializer='normal',
                  input_dim=14))
model.add(Dense(32,activation='relu'))
model.add(Dense(units=1,activation='linear',kernel_initializer='normal'))
model.summary()

#Adagrad = optimizers.Adagrad(lr=0.05, decay=0.0)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer= 'adam' , loss='mean_squared_error' , metrics=['mae','acc'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs',profile_batch=1)
checkpoint = ModelCheckpoint('last.h5', verbose=2, monitor='loss',save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', mode = "min",patience=50,restore_best_weights=True)
model.fit(X_train,y_train,batch_size=320,validation_data=(X_test,y_test),epochs=3000,verbose=2,callbacks=[tensorboard_callback,checkpoint,early_stopping])
pred = (model.predict(X_test),y_test)
s=model.predict_on_batch(X_test)

'''
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=256,activation="relu",early_stopping=True,random_state=1, max_iter=500,verbose=2).fit(X_train, y_train)
pa=clf.predict_proba(X_test[:1])
p=clf.predict(X_test)
s=clf.score(X_test, y_test)
'''