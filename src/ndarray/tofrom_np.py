import mxnet as mx
import numpy as np

a = np.array([1,2,3])
b = mx.nd.array(a)          

print b.asnumpy()    
