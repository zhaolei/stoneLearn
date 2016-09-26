from sklearn.datasets import make_moons, make_circles, make_classification
import pylab
import mstone
import csv

X,y= make_moons(5000, noise=0.4)

pylab.scatter(X[:,0], X[:,1], s=40, c=y, cmap=pylab.cm.Spectral)
