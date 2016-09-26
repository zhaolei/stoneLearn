
import mxnet as mx
import numpy as np
import mstone
import logging

head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
# load mxnet data iter
ita = mx.io.CSVIter(
    data_csv = '%smoon_data_x.csv'%mstone.data_path,
    data_shape =(2,),
    label_csv = '%smoon_data_y.csv'%mstone.data_path,
    label_shape = (1,), 
    batch_size=250
    )

print(ita)
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2048)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 2048)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")

'''
conv1 = mx.symbol.Convolution(data=act2, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
'''
fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 4048)
act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")

fc4  = mx.symbol.FullyConnected(data = act3, name = 'fc4', num_hidden = 4048)
act4 = mx.symbol.Activation(data = fc4, name='relu4', act_type="relu")
'''

fc5  = mx.symbol.FullyConnected(data = act4, name = 'fc5', num_hidden = 1024)
act5 = mx.symbol.Activation(data = fc5, name='relu5', act_type="relu")

fc6  = mx.symbol.FullyConnected(data = act5, name = 'fc6', num_hidden = 1024)
act6 = mx.symbol.Activation(data = fc6, name='relu6', act_type="relu")

fc7  = mx.symbol.FullyConnected(data = act6, name = 'fc7', num_hidden = 1024)
act7 = mx.symbol.Activation(data = fc7, name='relu7', act_type="relu")

fc8  = mx.symbol.FullyConnected(data = act7, name = 'fc8', num_hidden = 1024)
act8 = mx.symbol.Activation(data = fc8, name='relu8', act_type="relu")
'''

fc9  = mx.symbol.FullyConnected(data = act4, name='fc9', num_hidden=2)
mlp  = mx.symbol.SoftmaxOutput(data = fc9, name = 'softmax')


batch_size=60
batch_end_callback = []
batch_end_callback.append(mx.callback.Speedometer(batch_size, 50))

eval_metrics = ['accuracy']
## TopKAccuracy only allows top_k > 1
for top_k in [5, 10, 20]:
    eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

model = mx.model.FeedForward(
    ctx = mx.gpu(),
    symbol = mlp,
    num_epoch = 10,
    learning_rate = .01)


model.fit(
    X=ita,
    eval_metric        = eval_metrics,
    batch_end_callback = batch_end_callback)

y = model.predict(X=ita, return_data=True)

print(len(y))
print(len(y[0]))
print(len(y[0][0]))
for i in range(2):
    #pp = model.predict(y[1][1:10])
    #print(pp)
    pred = np.argsort(y[0][i])[::-1]
    print('-----------')
    print(pred)
    print(y[2][i])
