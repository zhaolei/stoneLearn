import mxnet as mx
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

trX = np.linspace(-1, 1, 10000)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
print trX.shape
print trY

X = mx.sym.Variable('data')
Y = mx.sym.Variable('softmax_label')

#Y_ = mx.sym.FullyConnected(data=X, num_hidden=1)

Y_ = mx.sym.FullyConnected(data=X, num_hidden=512, name='Y_')
act1 = mx.symbol.Activation(data = Y_, name='act1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 512)
act2 = mx.symbol.Activation(data = fc2, name='act2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 1)

cost = mx.sym.LinearRegressionOutput(data=fc3, label=Y)

model = mx.model.FeedForward(ctx = mx.gpu(0),symbol=cost, num_epoch=100, learning_rate=0.05, numpy_batch_size=1)

model.fit(X=trX, 
        y=trY,
        batch_end_callback = mx.callback.Speedometer(100, 200))
