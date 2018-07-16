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
import base
import sys

from sklearn import metrics

def Xd(alld, day=4):
    maxdd = np.max(alld[:,(0,1,2,3)])
    maxvv = np.max(alld[:,4])

    X = []
    Y = []
    D = []
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
            D.append(alld[i][5])

    X.append(tmx)

    X = np.array(X)
    X = X.reshape(X.shape[0],1, 16)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],1)
    return X,Y,D

def getDb(cc):
    dsql = "select open,high,low,close,num,datex  from stock where name='%s' order by datex asc"%cc
    base.cursor.execute(dsql)
    results = base.cursor.fetchall()
    hd = np.array(results,dtype='float') 
    return hd


if len(sys.argv) > 1:
    cco = sys.argv[1]
    alls = [cco]
else:
    alls = base.getList()
    
for c in alls:
    n = 0 
    model = load_model('/ds/model/stock/ls_%s_%d.h5'%(c,n))
    alld = getDb(c)

    X,Y,D = Xd(alld)
    print('code : %s'%c)

    xx = X[-20:]
    yy = Y[-19:]

    yp = model.predict(xx)
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    dd = D[-19:]
    print("error: %f"%np.sum(np.abs(yp[:-1]-yy)))
    print("date     : predict    true")
    for da in range(19):
        ss = "%d : %f    %f "%(dd[da], yp[da], yy[da])   
        print(ss)

    print("%s :   %f "%('NEXT',  yp[19]))   

