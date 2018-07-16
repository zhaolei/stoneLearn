'''
修正拆分之前 归一化 使用未来数据问题
数据来源 数据库 mysql
keras api更新到最新
日期: 2017-11-27
状态: 

'''
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
import keras
import time
import numpy as np
import pymysql
import datetime

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

wd = datetime.date.today().weekday()

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

def Xd(alld, day=4):
    maxdd = np.max(alld[:,(0,1,2,3)])
    maxvv = np.max(alld[:,4])

    X = []
    Y = []
    tmx = []
    for i in range(alld.shape[0]):
        s = np.array([])
        s = np.append(s, alld[i,2]/maxdd) 
        s = np.append(s, (alld[i,1] - alld[i,2])/maxdd) 
        s = np.append(s, (alld[i,0] - alld[i,3])/maxdd) 
        s = np.append(s, alld[i,4]/maxvv)

        tmx.append(s)
        if i >= day:
            X.append(tmx[:4])
            j = alld[i,3] - alld[i,0]
            if j > 0:
                Y.append(1.)
            else:
                Y.append(0.)
            tmx = tmx[1:]

    #X.append(tmx)
    X = np.array(X)
    X = X.reshape(X.shape[0],1, 16)

    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],1)

    return X,Y

def getDb(cc):
    dsql = "select open,high,low,close,num from stock where name='%s' order by datex asc"%cc
    cursor.execute(dsql)
    results = cursor.fetchall()
    hd = np.array(results,dtype='float') 
    return hd
    
allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
alls = allx.split(',')

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape = (None,layers[0]),
        units=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))


    model.add(LSTM(
        input_shape = (layers[1], 1024),
        units=1024,
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        1024,
        return_sequences=False))
    model.add(Dropout(0.4))


    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    
    return model


model = build_model([16, 512,4, 1])

for c in alls:
    alld = getDb(c)

    X,Y = Xd(alld)
    model.fit(X,
        Y,
        validation_data=(X[-20:],Y[-20:]),
        batch_size=100,
        epochs=6000)

    model.save('/ds/model/stock/ls_%s_%d.h5'%(c, wd))
    yp = model.predict(X[-20:])
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    print(yp.reshape(1,20).tolist())
    print(Y[-20:].reshape(1,20).tolist())

