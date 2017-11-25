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
'''
import pytz
import time
import datetime
tz = pytz.timezone('America/New_York')
a = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
print(a)
'''

def getNx():
    td = datetime.date.today()
    wd = datetime.date.today().weekday()

    d = 1
    if wd == 5:
        d = 2
    elif wd == 4:
        d = 3 
        
    ads = datetime.timedelta(days=d)
    rr = td + ads
    return rr.strftime('%Y-%m-%d')

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

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
    cursor.execute(dsql)
    results = cursor.fetchall()
    hd = np.array(results,dtype='float') 
    return hd
    
allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
alls = allx.split(',')
#stls = '''AAPL  ADBE  AMZN  ATVI  BIDU  EA  IBKR  INTC  KO  MKC  MSFT  MU  NFLX  NKE  NVDA  ORCL  SBUX  WMT'''
    #Ys.append(Y)
    #Xs.append(Xd(alld))


#fh = pd.read_pickle('bidu.d')
#alld = fh.values

#'Open', 'High', 'Low', 'Close', 'Volume',
#print(fh.columns)
#print(alld.shape)

'''
Y = alld[:,3] - alld[:,0]
Y = Y.reshape(Y.shape[0],1)
Y[Y>0.] = 1
Y[Y<=0.] = 0 
Y = Y[5:]
'''

#Y = keras.utils.to_categorical(Y, num_classes=2)

#X1 = X[:,:,(0,1,1)] - X[:,:,(2,3,0)]
#print(X[:,:,4].reshape(X[:,:,4].shape[0],1,1))
#X1 = np.column_stack((X1, X[:,:,4].reshape(X[:,:,4].shape[0],1,1)))
#print(X1.shape)
#print(X1[0])


#for i in range(60):
#    X[:,:,i] = (X[:,:,i] - X[:,:,i].min() )/ (X[:,:,i].max() - X[:,:,i].min())


for c in alls:
    n = 0 
    model = load_model('/ds/model/stock/ls_%s_%d.h5'%(c,n))
    alld = getDb(c)
    #ppt = '/ds/datas/stock/%s'%c
    #fh = pd.read_pickle(ppt)
    #alld = fh.values

    X,Y,D = Xd(alld)
    print(c)
    print(X.shape)
    print(Y.shape)

    xx = X[-20:]
    yy = Y[-19:]

    yp = model.predict(xx)
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    dd = D[-19:]
    for da in range(19):
        ss = "%d : %f    %f "%(dd[da], yp[da], yy[da])   
        print(ss)

    print("%s :   %f "%('NEXT',  yp[19]))   
    exit()
    #print(D[-19:])
    #print(yp.reshape(1,20).tolist())
    #print(yy.reshape(1,19).tolist())

