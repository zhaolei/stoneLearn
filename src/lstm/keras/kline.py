import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('/ds/github/stoneLearn/src/stock/')

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from keras.models import load_model

from klstm import build_model  

import logging
logging.basicConfig(level=logging.DEBUG)

x1 = np.linspace(1,1000,10000)

y1 = 3 * np.sin(x1 * 2) + 1

yy = y1.reshape(y1.shape[0],1)
xx = x1.reshape(x1.shape[0],1,1)


model = build_model([1, 256,128, 1])
model.fit(
    xx, 
    yy,
    batch_size=320,
    nb_epoch=10200)
#model.save('/ds/model/stock/jd.h5')  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')


'''
print(type(model))
predicted = regressor.predict(xdata['test'])

pp = [x for x in predicted]
pp = np.array(pp)
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
'''
