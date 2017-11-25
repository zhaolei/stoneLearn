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
from keras.models import load_model

from sklearn import metrics

wd = datetime.date.today().weekday()

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

def Xd(alld):
    maxdd = np.max(alld[:,(0,1,2,3)])
    maxvv = np.max(alld[:,4])

    X = []
    tmx = []
    for i in range(alld.shape[0]):
        s = np.array([])
        s = np.append(s, alld[i,2]/maxdd) 
        s = np.append(s, (alld[i,1] - alld[i,2])/maxdd) 
        s = np.append(s, (alld[i,0] - alld[i,3])/maxdd) 
        s = np.append(s, alld[i,4]/maxvv)

        tmx.append(s)
        if i > 2:
            X.append(tmx[-4:])
            tmx = tmx[-4:]

    X = X[:-1]
    X = np.array(X)
    X = X.reshape(X.shape[0],1, 16)
    return X

def getDb(cc):
    dsql = "select open,high,low,close,num from stock where name='%s' order by datex asc"%cc
    cursor.execute(dsql)
    results = cursor.fetchall()
    hd = np.array(results,dtype='float') 
    return hd
    
#allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
#alls = allx.split(',')

c = 'BIDU'
n = 0
model = load_model('/ds/model/stock/ls_%s_%d.h5'%(c,n))
alld = getDb(c)
Y = alld[:,3] - alld[:,0]
Y = Y.reshape(Y.shape[0],1)
Y[Y>0.] = 1
Y[Y<=0.] = 0 
Y = Y[4:]
X = Xd(alld)

'''
    model.fit(X,
    Y,
    validation_data=(X[-20:],Y[-20:]),
    batch_size=100,
    epochs=4000)

model.save('/ds/model/stock/ls_%s_%d.h5'%(c, wd))
'''
yp = model.predict(X[-19:])
yp[yp>=0.5] = 1.
yp[yp<0.5] = 0. 
print(yp.reshape(1,19).tolist())
print(Y[-20:].reshape(1,20).tolist())

