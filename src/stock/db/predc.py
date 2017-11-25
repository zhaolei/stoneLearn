import slist
import pymysql
import keras
import numpy as np
from keras.models import load_model

alls = slist.listcc

def getM(cc):
    n = 0
    model = load_model('/ds/model/stock/ls_%s_%d.h5'%(cc,n))
    return model

db = pymysql.connect("localhost","root","root","stone" )
cursor = db.cursor()

def getD(cc):
    dsql = "SELECT max(open+0.0) , max(high+0.0), max(low+0.0), max(close+0.0), max(num+0) FROM `stock` where name='%s'"
    cursor.execute(dsql%cc)
    results = cursor.fetchall()
    tx0 = np.array(results[0], dtype='float')
    maxdd = max(tx0[:3])    
    maxvv = tx0[4]    
    print(maxdd, maxvv)

    dsql = "SELECT open , high, low, close, num FROM `stock` where name='%s' order by datex asc limit 16 "
    cursor.execute(dsql%cc)
    results = cursor.fetchall()
    dm = np.array(results,dtype='float')

    Y = dm[:,0] - dm[:,3]
    Y[Y>0] = 1
    Y[Y<=.0] = 0 

    alld = dm
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

    #X = X[:-1]
    X = np.array(X)
    X = X.reshape(X.shape[0],1, 16)

    return X,Y[4:] 

for cc in alls:
    print("start -------%s"%cc)
    ym,Y = getD(cc)
    mo = getM(cc)
    rs = mo.predict(ym)
    rs[rs>=0.5] = 1
    rs[rs<0.5] = 0 
    print(Y)
    print(rs.reshape(rs.shape[0]))
    #print(mo)
    #print(ym.shape)
    #print(ym)

