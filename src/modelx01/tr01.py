import mxnet as mx
import numpy as np

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
softmax = mx.symbol.SoftmaxOutput(fc2, name='sm')

#x = np.array([1,2,3,4,5])
x = np.arange(3)
#x = np.array([1,2,3,4,5])
x = x.reshape(x.shape[0],1)
print x.shape
data_x = mx.nd.array(x)
print data_x.asnumpy()    

#y = np.arange(10)
y = np.array([[1],[2],[3]])
data_y = mx.nd.array(y)
print data_y.asnumpy()

batch_size = 3    
train_iter = mx.io.NDArrayIter(data_x, data_y, batch_size=batch_size, shuffle=True)
print train_iter
# create a model
num_epoch = 20    
#model = mx.model.FeedForward.create(
model = mx.model.FeedForward(
     softmax,
     num_epoch=num_epoch,
     learning_rate=0.01)

#model.fit(X=data_x, y=data_y)
model.fit(X=train_iter)

