import mxnet as mx
a = mx.nd.zeros((100, 50))

b = mx.nd.ones((256, 32, 128, 1))    

c = mx.nd.array([[1, 2, 3], [4, 5, 6]])    

print a.asnumpy()   

