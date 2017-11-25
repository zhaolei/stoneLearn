from lstm import  build_model
import tushare as ts
import numpy as np
from sklearn.metrics import mean_squared_error

tdata = ts.get_hist_data('600848',ktype='d')

nday = 2
fsdata = tdata.values[:-nday]
xdata = {}
xdata['train'] = fsdata[:300]
xdata['test'] = fsdata[300:380]
xdata['val'] = fsdata[380:]
    
ydata = {}
fclose = [w for w in tdata.close]
fclose = np.array(fclose)
fclose = fclose[nday:]
print(fsdata.shape)
print(fclose.shape)
ydata['train'] = fclose[:300]
ydata['test'] = fclose[300:380]
ydata['val'] = fclose[380:]

xdata['train'] = xdata['train'].reshape(xdata['train'].shape[0],1,xdata['train'].shape[1])
xdata['test'] = xdata['test'].reshape(xdata['test'].shape[0],1,xdata['test'].shape[1])
xdata['val'] = xdata['val'].reshape(xdata['val'].shape[0],1,xdata['val'].shape[1])
ydata['train'] = ydata['train'].reshape(ydata['train'].shape[0],1)
ydata['test'] = ydata['test'].reshape(ydata['test'].shape[0],1)
ydata['val'] = ydata['val'].reshape(ydata['val'].shape[0],1)

xdata['train'] = xdata['train'].astype(np.float32)
xdata['test'] = xdata['test'].astype(np.float32)
xdata['val'] = xdata['val'].astype(np.float32)
ydata['train'] = ydata['train'].astype(np.float32)
ydata['test'] = ydata['test'].astype(np.float32)
ydata['val'] = ydata['val'].astype(np.float32)


print(xdata['train'].shape)
model = build_model([14, 10,10, 1])
model.fit(xdata['train'], ydata['train'],batch_size=10,nb_epoch=40)

predicted = model.predict(xdata['test'])

pp = [x for x in predicted]
pp = np.array(pp)
rmse = np.sqrt(((pp - ydata['test']) ** 2).mean(axis=0))
score = mean_squared_error(pp, ydata['test'])
print ("MSE: %f" % score)
