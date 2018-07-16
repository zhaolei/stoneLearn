'''
 归一化问题
在拆分 训练集和 验证集的之前 做了 归一化 使用了未来的数据 包括未来时间段的最大或者最小

状态: 放弃版本

'''
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from keras import regularizers

import keras
import time
import numpy as np
import pymysql
import datetime
import db.base

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

wd = datetime.date.today().weekday()

#print(db.base)
#db = pymysql.connect("localhost","root","root","stone" )
#cursor = db.cursor()

def Xd(alld, day=4):
    maxdd = np.max(alld[:,(0,1,2,3)])
    maxvv = np.max(alld[:,4])
    scaler.fit(alld[:,(0,1,2,3)])
    X1 = scaler.transform(alld[:,(0,1,2,3)])
    scaler.fit(alld[:,(4)])
    X2 = scaler.transform(alld[:,(4,)])
    print(X1.shape)
    print(X2.shape)
    Xa = np.hstack((X1,X2))

    X = []
    Y = []
    for i in range(Xa.shape[0]):
        if i >= day:
            X.append(Xa[i-4:i])
            Y.append(alld[i,3] - alld[i,0])

    #X.append(tmx)
    X = np.array(X)
    print(X.shape)
    X = X.reshape(X.shape[0],1, 20)

    Y = np.array(Y)
    print(Y.shape)
    Y = Y.reshape(Y.shape[0],1)
    Y[Y>0] = 1
    Y[Y<=0] =0 

    return X,Y

def getDb(cc):
    dsql = "select open,high,low,close,num from stock where name='%s' order by datex asc"%cc
    db.base.cursor.execute(dsql)
    results = db.base.cursor.fetchall()
    hd = np.array(results,dtype='float') 
    return hd
    
#allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
#alls = allx.split(',')
alls = db.base.getList()

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape = (None,layers[0]),
        units=layers[1],
        kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l2(0.01),
        return_sequences=True))
    model.add(Dropout(0.4))
    '''
    model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
    '''


    model.add(LSTM(
        input_shape = (layers[1], 1024),
        units=1024,
        kernel_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l2(0.01),
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        1024,
        return_sequences=False))
    model.add(Dropout(0.4))

    '''


    model.add(LSTM(
        input_dim=1024,
        output_dim=1024,
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        1024,
        return_sequences=False))
    model.add(Dropout(0.4))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))
    '''

    '''
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              #optimizer="rmsprop",
              metrics=['accuracy'])

    '''
    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model


model = build_model([20, 512,4, 1])

for c in alls:
    alld = getDb(c)
    
    all_setX, all_setY = Xd(alld)

    X,Y = all_setX[:-14], all_setY[:-14]
    valX,valY = all_setX[-14:], all_setY[-14:]
    model.fit(X,
        Y,
        validation_data=(valX,valY),
        batch_size=220,
        epochs=8000)

    model.save('/ds/model/stock/ls_%s_%d_d.h5'%(c, wd))
    yp = model.predict(valX)
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    pp = yp.reshape(-1,).tolist()
    vpp = valY.reshape(-1,).tolist()
    jtp = yp - valY
    jtp[jtp<0.0] = 1
    jpp = jtp.reshape(-1,).tolist()
    print(pp, file=open('1213.log','a'))
    print(vpp, file=open('1213.log','a'))
    print('error: %d'%sum(jpp), file=open('1213.log','a'))

