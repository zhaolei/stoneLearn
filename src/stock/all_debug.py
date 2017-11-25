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
    
allx = 'AAPL,ADBE,AMZN,ATVI,BIDU,EA,IBKR,INTC,KO,MKC,MSFT,MU,NFLX,NKE,NVDA,ORCL,SBUX,WMT,FB'
alls = allx.split(',')
#print(fh.columns)

#Y = keras.utils.to_categorical(Y, num_classes=2)
#X1 = X[:,:,(0,1,1)] - X[:,:,(2,3,0)]
#print(X[:,:,4].reshape(X[:,:,4].shape[0],1,1))
#X1 = np.column_stack((X1, X[:,:,4].reshape(X[:,:,4].shape[0],1,1)))
#print(X1.shape)
#print(X1[0])


#for i in range(60):
#    X[:,:,i] = (X[:,:,i] - X[:,:,i].min() )/ (X[:,:,i].max() - X[:,:,i].min())


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))


    model.add(LSTM(
        input_dim=layers[1],
        output_dim=1024,
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
        output_dim=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)
    return model


model = build_model([16, 512,4, 1])

for c in alls:
    alld = getDb(c)
    #ppt = '/ds/datas/stock/%s'%c
    #fh = pd.read_pickle(ppt)
    #alld = fh.values
    Y = alld[:,3] - alld[:,0]
    Y = Y.reshape(Y.shape[0],1)
    Y[Y>0.] = 1
    Y[Y<=0.] = 0 
    Y = Y[4:]

    X = Xd(alld)
    model.fit(X,
        Y,
        validation_data=(X[-20:],Y[-20:]),
        batch_size=100,
        epochs=4000)

    model.save('/ds/model/stock/ls_%s_%d.h5'%(c, wd))
    yp = model.predict(X[-20:])
    yp[yp>=0.5] = 1.
    yp[yp<0.5] = 0. 
    print(yp.reshape(1,20).tolist())
    print(Y[-20:].reshape(1,20).tolist())

