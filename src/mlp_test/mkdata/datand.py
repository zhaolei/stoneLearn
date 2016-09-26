from sklearn.datasets import make_moons, make_circles, make_classification
import pylab
import mstone
import csv

#data_0 = make_moons(150000, noise=0.1)
num=15000
feature = 4
class_n = 2

X,y = make_classification(num, feature, class_n)




xdx = csv.writer(open('%s/nd_data_x.csv'%mstone.data_path, 'w'))
xlab = csv.writer(open('%s/nd_data_y.csv'%mstone.data_path, 'w'))


for x in X:
    xdx.writerow(x)

for x in y:
    xlab.writerow([x])





