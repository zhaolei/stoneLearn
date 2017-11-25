import numpy as np
from numpy import genfromtxt
#import mstone
 
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
 
    return 1/(1+np.exp(-x))



'''
dataX = genfromtxt('%s/smlp_data_x.csv'%mstone.data_path, delimiter=',')
dataY = genfromtxt('%s/smlp_data_y.csv'%mstone.data_path, delimiter=',')
dataY = dataY.reshape(dataY.shape[0],1)
print dataY

'''
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
 
y = np.array([[0],
            [1],
            [1],
            [0]])

dataX = X
dataY = y
print dataY.shape
print dataX.shape

 
np.random.seed(1)
 
# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,108)) - 1
syn1 = 2*np.random.random((108,102)) - 1
syn2 = 2*np.random.random((102,1)) - 1
 
print 'syn0.shape %d' , syn0.shape
print 'syn1.shape %d' , syn1.shape

print 'X.shape %d' , dataX.shape
print 'Y.shape %d' , dataY.shape

alps = 0.05
lview = []
for j in xrange(40000):
 
    # Feed forward through layers 0, 1, and 2
    l0 = dataX
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l3 = nonlin(np.dot(l2,syn2))
 
    # how much did we miss the target value?
    #l2_error = y - l2
    l3_error = dataY -  l3
 
    if (j% 100) == 0:
        #print "Error:" + str(np.mean(np.abs(l2_error)))
        print "Error:[" + str(j) + "]" + str(np.mean(np.abs(l3_error)))
        #print 'syn0',syn0
        #print 'syn1',syn1
        
    '''
    if(j%100) == 0:
        lview.append(l3_error)
    '''
 
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l3_delta = l3_error*alps*nonlin(l3, deriv=True)
    
    l2_error = l3_delta.dot(syn2.T)
    #l2_error = l3_delta.dot(syn2)
    
    l2_delta = l2_error*alps*nonlin(l2,deriv=True)
 
    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
 
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * alps * nonlin(l1,deriv=True)

 
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print l3
  
