import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import pylab 

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

oydata = tdata.Close.values - tdata.Open.values 

oxdata = oydata.reshape(oydata.shape[0], 1,1)
txdata = oxdata[50:]
tydata = oydata[50:]

vxdata = oxdata[:50]
vydata = oydata[:50]



model = build_model([1, 256,128, 1])
model.fit(
    txdata, 
    tydata,
    batch_size=180,
    nb_epoch=200)

vv = model.predict(vxdata)
'''
print(vv)
print(vv - vydata)
print(sum(vv - vydata))
'''

pylab.plot(vv)
pylab.plot(vydata)
pylab.show()
