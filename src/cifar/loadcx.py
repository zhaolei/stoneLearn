import cPickle
import mstone
import loadcifar

cfile = '%s/cifar-10-batches-py/batches.meta'%mstone.data_path
print cfile
fo = open(cfile, 'rb')
dx = cPickle.load(fo)
print dx

w = loadcifar.cifar10()
print type(w[1][0])
for si in w[1]:
    print si
    #print vx
