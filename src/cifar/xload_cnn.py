
import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from convolutional_mlp import LeNetConvPoolLayer 
from mlp import HiddenLayer

import mstone
import cPickle
from sklearn import metrics

def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average="macro")
    m_recall = metrics.recall_score(actual, pred, average="macro")
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)


#model = '%s/save_model.pkl'%mstone.theano_path
#model = '/ds/model/cifar100/best_model_c.pkl'
model = '%s/save_model_cifar100.pkl'%mstone.theano_path
testset = '/ds/datas/mnist.pkl.gz'
batch_size = 1000
#classifiers = cPickle.load(gzip.open(model))
classifiers = cPickle.load(open(model))
print type(classifiers[0]) 

      # Pick out the individual layer
layer0_input = classifiers[0]
layer0 = classifiers[1]
layer1 = classifiers[2]
layer2_input = classifiers[3]
layer2 = classifiers[4]
layer3 = classifiers[5]

# Apply it to our test set
testsets = load_data(testset)
test_set_x,test_set_y = testsets[2]
#test_set_x = testsets.get_value()
test_set_x = test_set_x.get_value()

# compile a predictor function
index = T.lscalar()

predict_model = theano.function(
    [layer0_input],
    layer3.y_pred,
)

print test_set_x.shape
print type(test_set_x)
ix = 5 
#predicted_values = predict_model(
#    test_set_x[ix * batch_size:(ix + 1) * batch_size].reshape((batch_size, 1, 28, 28))
#)

#
testt = test_set_x[:600]
predicted_values = predict_model(
    #test_set_x.reshape((test_set_x.shape[0], 1, 32, 32))
    testt.reshape((testt.shape[0], 1, 32, 32))
)

#print('Prediction complete.')
#print(predicted_values)
test_y = test_set_y.owner.inputs[0].owner.inputs[0].get_value()
#print dir(test_set_y.owner.inputs[0].get_value)
#or_y = numpy.asarray(test_y[ix * batch_size:(ix + 1)*batch_size],dtype='int32')
or_y = numpy.asarray(test_y, dtype='int32')
yy = or_y[:600]
print predicted_values
print yy 

evaluate(yy, predicted_values)
