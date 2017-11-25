import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('/ds/github/stoneLearn/src/stock/')

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from klstm import build_model  
from sklearn.datasets import make_moons, make_circles, make_classification

import logging
logging.basicConfig(level=logging.DEBUG)

X,y= make_moons(1000, noise=0.1)

X =X.reshape(X.shape[0],1,X.shape[1])


model = build_model([2, 256,128, 1])
model.fit(
    X, 
    y,
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
