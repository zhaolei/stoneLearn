
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


layer0 = pickle.load(open('%s/best_model_cnn_l0.pkl'%mstone.theano_path))
layer1 = pickle.load(open('%s/best_model_cnn_l1.pkl'%mstone.theano_path))
layer2 = pickle.load(open('%s/best_model_cnn_l2.pkl'%mstone.theano_path))
layer3 = pickle.load(open('%s/best_model_cnn_l3.pkl'%mstone.theano_path))


layer0_model = theano.function(
    inputs=[layer0.input],
    outputs=layer0.output)

layer1_model = theano.function(
    inputs=[layer1.input],
    outputs=layer1.output)

layer2_model = theano.function(
    inputs=[layer2.input],
    outputs=layer2.output)

layer3_model = theano.function(
    inputs=[layer3.input],
    outputs=layer3.y_pred)

layerx_model = theano.function(
    inputs=[layer0.input],
    outputs=layer3.y_pred)

'''
test_model = theano.function(
    inputs=[index],
    outputs=layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
'''

dataset='/ds/datas/mnist.pkl.gz'
datasets = load_data(dataset)
test_set_x, test_set_y = datasets[2]
train_set_x, train_set_y = datasets[0]


index = T.lscalar()
batch_size = 100
x = T.matrix('x')
'''
layerp_predict = theano.function(
    [index],
    layer3.y_pred,
    givens={
        x:test_set_x[index * batch_size : (index+1)*batch_size]
    }
)
'''





#test_set_x = test_set_x.get_value()
trainv_set_x = train_set_x.get_value()

#input_test = test_set_x[:20].reshape((20,1,28,28))
input_test = trainv_set_x[20].reshape((1,1,28,28))
#input_test = test_set_x[210].reshape((1,1,28,28))
print input_test.shape
#predicted_values = predict_model(test_set_x[:10])
#print(predicted_values)
'''
data_l0 = layer0_model(input_test)
print data_l0.shape
data_l1 = layer1_model(data_l0)
print data_l1.shape

data_l2_in = data_l1.flatten(2)
data_l2_in = data_l2_in.reshape(1,800)
print data_l2_in.shape
#print data_l2_in.shape
#data_l2 = layer2_model(data_l1)
data_l2 = layer2_model(data_l2_in)
print data_l2.shape
data_l3 = layer3_model(data_l2)
print data_l3.shape
print data_l3
#print dir(test_set_y.owner.inputs[0].owner.inputs[0])
sy = train_set_y.owner.inputs[0].owner.inputs[0].get_value()
print sy[20]
#sok = numpy.array(sy[:20],dtype='int32')
#print sok==data_l3

#print data_l1.shape
'''

'''
index = T.lscalar()
batch_size = 100
x = T.matrix('x')
y = T.ivector('y')  # the labels are presented as 1D vector of
test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    },
    on_unused_input='ignore'
)
layerp_predict = theano.function(
    [index],
    layer3.y_pred,
    givens={
        x:test_set_x[0 : 100]
    },
    on_unused_input='ignore'
)
#print dir(layer3)
#print dir(layer3.y_pred)
model_predict = theano.function([index], layer3.y_pred,
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size]})

'''
