import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
import keras
import time
import numpy as np

from sklearn import metrics

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


stls = '''AAPL  ADBE  AMZN  ATVI  BIDU  DD  EA  IBKR  INTC  KO  MKC  MSFT  MU  NFLX  NKE  NVDA  ORCL  SBUX  WMT'''
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

for c in stls.split('  '):
    ppt = '/ds/datas/stock/%s'%c
    fh = pd.read_pickle(ppt)
    alld = fh.values
    Y = alld[:,3] - alld[:,0]
    Y = Y.reshape(Y.shape[0],1)
    Y[Y>0.] = 1
    Y[Y<=0.] = 0 
    Y = Y[4:]

    X = Xd(alld)
    model.fit(X,
        Y,
        validation_data=(X[-100:-50],Y[-100:-50]),
        batch_size=100,
        epochs=4000)

    model.save('/ds/model/stock/ls_%s.h5'%c)

#model.save('/ds/model/stock/bidux.h5')
yp = model.predict(X[-50:])

yp[yp>=0.5] = 1.
yp[yp<0.5] = 0. 
print(yp)
print(Y[-20:] - yp)
fpr, tpr, thresholds = metrics.roc_curve(Y[-20:].reshape([20,]), yp.reshape([20,]),  pos_label=1.)
print(fpr)
print(tpr)
print(metrics.auc(fpr, tpr))
fpr, tpr, thresholds = metrics.roc_curve(Y[-20:].reshape([20,]), yp.reshape([20,]),  pos_label=0.)
print(fpr)
print(tpr)
print(metrics.auc(fpr, tpr))
