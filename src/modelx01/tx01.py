import mxnet as mx
import numpy
import sys

#print sys.argv
args = sys.argv[1:]
try:
    print args
    n_layers = int(args[0])
except:
    print 'The first argument should be number of layers!'
    quit()

nn = None
try:
    nn = mx.symbol.Variable('data')
    print 'shape is', nn.infer_shape
except:
    print 'Failed to initialize input layer!'
    quit()

try:
    for i in range(2, n_layers):
        nn = mx.symbol.FullyConnected(data = nn, name = 'hidden-recept-' + str(i), num_hidden = int(args[i]))
        nn = mx.symbol.Activation(data = nn, name = 'hidden-act-' + str(i), act_type = 'relu')
except:
    print 'Failed to initailize hidden layer!'
    quit()

try:
    nn = mx.symbol.FullyConnected(data = nn, name = 'hidden-recept-' + str(n_layers), num_hidden = int(args[n_layers]))
    nn = mx.symbol.LinearRegressionOutput(data = nn, name = 'softmax')
except:
    print 'Failed to build output layer!'
    quit()

data = open('data').readlines()
x,y = [],[]
for i in xrange(len(data)):
    if i % 2 == 0:
        x.append(map(float, data[i].split()))
    else:
        y.append(map(float, data[i].split()))
x = mx.ndarray.array(x).asnumpy()
y = mx.ndarray.array(y).asnumpy()
iterator = mx.io.NDArrayIter(data = x, label = y, batch_size = 128)
#print iterator.provide_data, iterator.provide_label

data = open('test').readlines()
x,y = [],[]
for i in xrange(len(data)):
    if i % 2 == 0:
        x.append(map(float, data[i].split()))
    else:
        y.append(map(float, data[i].split()))
x = mx.ndarray.array(x).asnumpy()
y = mx.ndarray.array(y).asnumpy()
tester = mx.io.NDArrayIter(data = x, label = y, batch_size = 128)
#print provide_data, provide_label

model = mx.model.FeedForward(
        symbol = nn,
        ctx = mx.gpu(0),
        num_epoch = 1000,
        learning_rate = .1,
        momentum = .9,
        wd = 1e-5,
        initializer = mx.init.Xavier(factor_type = 'in', magnitude = 2.34)
)

print iterator.provide_data, iterator.provide_label
model.fit(
        X = iterator,
        eval_data = tester,
        batch_end_callback = mx.callback.Speedometer(128, 50)
)

model.save("res")
tester.next()
print model.predict(tester)
