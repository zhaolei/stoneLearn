'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import datasets
import random
import mstone
import csv

fname = "%s/moon_data_x.csv"%mstone.data_path
reader=csv.reader(open(fname,"r"),delimiter=',')
xdata=list(reader)
#xdata=np.array(x).astype('float')

fname = "%s/moon_data_y.csv"%mstone.data_path
reader=csv.reader(open(fname,"r"),delimiter=',')
ydata=list(reader)
ydata=np.array(ydata).astype('int32')

#r = np.genfromtxt(fname, delimiter=',',dtype=None, names=True)
#r = np.loadtxt(open(fname,'r'), delimiter=',')


X_train = []
Y_train = []

X_test = []
Y_test = []

for x in range(len(xdata)):
    if int(random.random() * 100) % 7 != 2: 
        X_train.append(xdata[x])
        Y_train.append(ydata[x])
    else:
        X_test.append(xdata[x])
        Y_test.append(ydata[x])


print(len(X_train))
print(len(X_train[0]))
print(len(Y_train))
print(Y_train[0])

batch_size = 128
nb_classes = 2 
nb_epoch = 200

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#(X_train, y_train), (X_test, y_test),(X_valid, y_valid) = mnist.load_data(path='/ds/datas/mnist.pkl.gz')
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape)
print(X_test.shape)

'''
X_train = X_train.reshape(50000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
'''
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
