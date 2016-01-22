import mxnet as mx
a = mx.nd.zeros((100, 50), mx.gpu(0))

print a.asnumpy()   

