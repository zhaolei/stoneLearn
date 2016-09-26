from sklearn.datasets import make_moons, make_circles, make_classification
import pylab
import mstone
import csv

data_0 = make_moons(150000, noise=0.1)

xdx = csv.writer(open('%s/moon_data_x.csv'%mstone.data_path, 'w'))
xlab = csv.writer(open('%s/moon_data_y.csv'%mstone.data_path, 'w'))


for x in data_0[0]:
    xdx.writerow(x)

for x in data_0[1]:
    xlab.writerow([x])





