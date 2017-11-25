import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm import generate_data, lstm_model

import logging
logging.basicConfig(level=logging.DEBUG)

LOG_DIR = '/tmp/m/ops_logs/6/ts'
TIMESTEPS =  13 
RNN_LAYERS = [{'num_units': 32},{'num_units': 64},{'num_units': 64}]
DENSE_LAYERS = None
TRAINING_STEPS = 40000
PRINT_STEPS = TRAINING_STEPS / 100
BATCH_SIZE = 20

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))


X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)
'''
tushare data
'''
import tushare as ts
tdata = ts.get_hist_data('600848',ktype='d')

'''
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
fsdata = tdata.values
xdata = {}
xdata['train'] = fsdata[:300]
xdata['test'] = fsdata[300:380]
xdata['val'] = fsdata[380:]
    
ydata = {}
ydata['train'] = fsdata[:300,0]
ydata['test'] = fsdata[300:380,0]
ydata['val'] = fsdata[380:,0]

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


print X['train'].shape
print X['train'][0]
print X['train'].dtype
print xdata['train'].shape
print xdata['train'][0]
print xdata['train'].dtype
print '**********'
print y['train'].shape
print y['train'][0]
print ydata['train'].shape
print ydata['train'][0]
#exit()


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

print 'ok'
predicted = regressor.predict(xdata['test'])

pp = [x for x in predicted]
pp = np.array(pp)
print type(predicted)
print type(y['test'])
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
