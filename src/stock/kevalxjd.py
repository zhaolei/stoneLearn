import numpy as np
import pandas as pd
import tensorflow as tf
import sys

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from klstm import build_model  
from stockd import get_data, get_local_data

import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) < 2:
    print('No code')
    exit()

xcode = sys.argv[1]
tdata = get_local_data(xcode)

nday = 1 

odata = tdata.values[:,(0,1,2,4,6,7,8,9,10,11)]
oxdata = tdata.values[:,3]

fsdata = odata[:-nday]
#fsdata = tdata.values
#guiyi
for i in range(fsdata.shape[1]):
    ksum = np.sqrt(sum(pow(fsdata[:,i],2)))
    print(" %d %d"%(i, ksum))
    fsdata[:,i] /= (ksum*1.0)


xtrain = int((oxdata.shape[0] / 10) * 8)
xtest = int((oxdata.shape[0] / 10) * 1)

print(fsdata.shape)
print(xtrain)
print(xtest)

xdata = {}
xdata['train'] = fsdata[:xtrain]
xdata['val'] = fsdata[xtrain:xtrain + xtest]
xdata['test'] = fsdata[xtrain+xtest:]
    
ydata = {}
fclose = [w for w in tdata.Close]
fclose = np.array(fclose)

fclose = fclose[nday:]
print(fsdata.shape)
print(fclose.shape)
ydata['train'] = fclose[:xtrain]
ydata['val'] = fclose[xtrain:xtrain+xtest]
ydata['test'] = fclose[xtrain + xtest:]

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

print(xdata['train'][0][0])
print(ydata['train'][0])

model = build_model([10, 256,128, 1])
model.fit(
    xdata['train'], 
    ydata['train'],
    batch_size=180,
    nb_epoch=20000,
    validation_data=(xdata['val'],ydata['val']))

model.save('/ds/model/stock/%s.h5'%xcode)  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')


print(type(model))
predicted = model.predict(xdata['test'])

pp = [x for x in predicted]
pp = np.array(pp)
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
