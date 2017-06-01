import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm import  lstm_model
from stockd import get_data, get_local_data

import logging
logging.basicConfig(level=logging.DEBUG)

import cPickle as picle

LOG_DIR = '/tmp/m/ops_logs/8/ts'
TIMESTEPS =  12 
RNN_LAYERS = [{'num_units': 32},{'num_units': 128},{'num_units': 64}]
DENSE_LAYERS = None
TRAINING_STEPS = 800
PRINT_STEPS = TRAINING_STEPS / 200
BATCH_SIZE = 64

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

xcode = 'JD'
tdata = get_local_data(xcode)

'''
tushare data
import tushare as ts
tdata = ts.get_hist_data('600848',ktype='d')

fdata = []
fdata.append([w for w in tdata.open])
fdata.append([w for w in tdata.close])
fdata.append([w for w in tdata.low])
fdata.append([w for w in tdata.high])
fdata.append([w for w in tdata.volume])
'''
#fdata = tdata.values

#fndata = np.array(fdata)
#fsdata = fndata.T

nday = 2
fsdata = tdata.values[:-nday]

xtrain = (tdata.values.shape[0] / 10) * 6
xtest = (tdata.values.shape[0] / 10) * 3

xdata = {}
xdata['train'] = fsdata[:xtrain]
xdata['test'] = fsdata[xtrain:xtrain + xtest]
xdata['val'] = fsdata[xtrain+xtest:]
    
ydata = {}
fclose = [w for w in tdata.Close]
fclose = np.array(fclose)
fclose = fclose[nday:]
print(fsdata.shape)
print(fclose.shape)
ydata['train'] = fclose[:xtrain]
ydata['test'] = fclose[xtrain:xtrain+xtest]
ydata['val'] = fclose[xtrain + xtest:]

xdata['train'] = xdata['train'].reshape(xdata['train'].shape[0],xdata['train'].shape[1],1)
xdata['test'] = xdata['test'].reshape(xdata['test'].shape[0],xdata['test'].shape[1],1)
xdata['val'] = xdata['val'].reshape(xdata['val'].shape[0],xdata['val'].shape[1],1)
ydata['train'] = ydata['train'].reshape(ydata['train'].shape[0],1)
ydata['test'] = ydata['test'].reshape(ydata['test'].shape[0],1)
ydata['val'] = ydata['val'].reshape(ydata['val'].shape[0],1)

xdata['train'] = xdata['train'].astype(np.float32)
xdata['test'] = xdata['test'].astype(np.float32)
xdata['val'] = xdata['val'].astype(np.float32)
ydata['train'] = ydata['train'].astype(np.float32)
ydata['test'] = ydata['test'].astype(np.float32)
ydata['val'] = ydata['val'].astype(np.float32)



# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(xdata['val'], ydata['val'],
                                                     every_n_steps=PRINT_STEPS)
#                                                     early_stopping_rounds=100)
# print(X['train'])
# print(y['train'])

regressor.fit(xdata['train'], ydata['train'], 
              monitors=[validation_monitor], 
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)

fff = regressor.get_params()
print(dir(fff))
print(type(fff))
print(fff)
fp = open('/ds/model/stock/jd.pkl', 'wb')
picle.dump(regressor.get_params(),fp)
fp.close()
print 'ok'
predicted = regressor.predict(xdata['test'])


pp = [x for x in predicted]
pp = np.array(pp)
print type(predicted)
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
