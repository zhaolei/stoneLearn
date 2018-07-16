import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
        input_dim=layers[1],
        output_dim=layers[2],
        return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
        input_dim=layers[2],
        output_dim=128,
        return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(
        128,
        return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


x1 = np.linspace(1,100,1000)
y1 = np.sin(x1)
xv1 = x1.reshape(x1.shape[0],1,1)
print(xv1.shape)
exit()
model = build_model([1, 4,4, 1])
model.fit(xv1,y1,batch_size=10,nb_epoch=40)
