import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('/ds/github/stoneLearn/src/stock/')

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from klstm import build_model  
from stockd import get_data, get_local_data

import logging
logging.basicConfig(level=logging.DEBUG)



xcode = 'DD'
tdata = get_local_data(xcode)

#fdata = tdata.values

#fndata = np.array(fdata)
#fsdata = fndata.T

nday = 1 
#fsdata = tdata.values[:-nday]
fsdata = tdata.values

xtrain = int((tdata.values.shape[0] / 10) * 7)
xtest = int((tdata.values.shape[0] / 10) * 2)

print(fsdata.shape)
print(xtrain)
print(xtest)

xdata = {}
xdata['train'] = fsdata[:xtrain]
xdata['test'] = fsdata[xtrain:xtrain + xtest]
xdata['val'] = fsdata[xtrain+xtest:]
    
ydata = {}
fclose = [w for w in tdata.Close]
fclose = np.array(fclose)
#fclose = fclose[nday:]
print(fsdata.shape)
print(fclose.shape)
ydata['train'] = fclose[:xtrain]
ydata['test'] = fclose[xtrain:xtrain+xtest]
ydata['val'] = fclose[xtrain + xtest:]

print(xdata['train'].shape)
xdata['train'] = xdata['train'].reshape(xdata['train'].shape[0],1,xdata['train'].shape[1])
xdata['test'] = xdata['test'].reshape(xdata['test'].shape[0],1,xdata['test'].shape[1])
xdata['val'] = xdata['val'].reshape(xdata['val'].shape[0],1,xdata['val'].shape[1])
ydata['train'] = ydata['train'].reshape(ydata['train'].shape[0],1)
ydata['test'] = ydata['test'].reshape(ydata['test'].shape[0],1)
ydata['val'] = ydata['val'].reshape(ydata['val'].shape[0],1)

xdata['train'] = xdata['train'].astype(np.float32)
xdata['test'] = xdata['test'].astype(np.float32)
xdata['val'] = xdata['val'].astype(np.float32)
ydata['train'] = ydata['train'].astype(np.float32)
ydata['test'] = ydata['test'].astype(np.float32)
ydata['val'] = ydata['val'].astype(np.float32)


ym = ydata['train']
xm = ydata['train'].reshape(ydata['train'].shape[0],1,1)

model = build_model([1, 256,128, 1])
model.fit(
    xm, 
    ym,
    batch_size=320,
    nb_epoch=2000,
    validation_data=(ydata['val'].reshape(ydata['val'].shape[0],1,1),ydata['val']))
model.save('/ds/model/stock/jd.h5')  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')


print(type(model))
'''
predicted = regressor.predict(xdata['test'])

pp = [x for x in predicted]
pp = np.array(pp)
print type(predicted)
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
'''
