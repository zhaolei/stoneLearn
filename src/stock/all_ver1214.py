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
import keras
import time
import numpy as np
import pymysql
import datetime
import pytz
from keras.models import load_model
import sys

import db.base

acc = 'check'

if len(sys.argv) > 1:
    acc = sys.argv[1]

wd = datetime.date.today().weekday()

tz = pytz.timezone('America/New_York')
dday = datetime.datetime.now(tz)

def getPreDay():
    if dday.hour > 9:
        nday = dday + datetime.timedelta(days=1)
        return nday.strftime("%Y%m%d")

    return dday.strftime("%Y%m%d")


def Xd(alld, day=4):
    maxdd = np.max(alld[:-20,(0,1,2,3)])
    maxvv = np.max(alld[:-20,4])

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

    lt = np.array(tmx)
    lt = lt.reshape(1,1,16)

    return X,Y,lt

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


predate = getPreDay()
logfile = '/ds/log/stock/%s.log'%predate
model = load_model('/ds/model/stock/ls_BASE_d12.h5')

epx = 0
for c in alls:
    epx += 1
    eps = 300    

    alld = getDb(c)

    X,Y,lt = Xd(alld)
    if acc == 'train':
        model.fit(X,
            Y,
            validation_data=(X[-20:],Y[-20:]),
            batch_size=200,
            epochs=eps)
        model.save('/ds/model/stock/ls_%s_d12.h5'%(c))
    else:
        model = load_model('/ds/model/stock/ls_%s_d12.h5'%(c))

    yp = model.predict(X[-20:])
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    pp = yp.reshape(-1,).tolist()
    vpp = Y[-20:].reshape(-1,).tolist()
    jtp = yp - Y[-20:] 
    jtp[jtp<0.0] = 1
    jpp = jtp.reshape(-1,).tolist()
    print('code : %s'%c, file=open(logfile,'a'))
    print(pp, file=open(logfile,'a'))
    print(vpp, file=open(logfile,'a'))
    print('error: %d'%sum(jpp), file=open(logfile,'a'))
    rt = model.predict(lt)
    print("debug***********", file=open(logfile,'a'))
    print(rt, file=open(logfile,'a'))
    nnd = int(time.time())
    psql = "INSERT INTO `stock_predict` (`name`, `datex`, `value`, `datetime`) VALUES ('%s', '%s', '%f', '%d');"%(c, predate, rt[0][0], nnd)
    db.base.cursor.execute(psql)
    db.base.db.commit()

