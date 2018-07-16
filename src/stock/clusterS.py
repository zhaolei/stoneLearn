import numpy as np
import pandas as pd
#import tensorflow as tf
import sys

#from tensorflow.contrib import learn
#from sklearn.metrics import mean_squared_error

#from klstm import build_model  
from stockd import get_data, get_local_data

import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) < 2:
    print('No code')
    exit()

def dis_f(a, b): 
    #a = np.array(a)
    #b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #return (np.linalg.norm(a) * np.linalg.norm(b))

xcode = sys.argv[1]
tdata = get_local_data(xcode)

nday = 1 

#odata = tdata.values[:,(0,1,2,3,4,7,8,9,10,11)]
odata = tdata.values[:,(0,1,3,2)]
#print(tdata.columns)
vdata = 100*((odata[:,(0,1)] - odata[:,(2,3)])/odata[:,(0,0)])

#'Open', 'High', 'Low', 'Close'
#print(vdata.shape)

#一个曲线几天
dy = int(sys.argv[2])

vlist = []
for v in range(dy, vdata.shape[0]):
    vlist.append(vdata[v-dy:v].reshape(1,2*dy)[0])


numC = 0
clist = {}
mlist=[]

DMAX = 0.5

for v in range(len(vlist)):
    for vx in range(v, len(vlist)):
        
        numC += 1
        #print("%d:%d"%(v, vx))
        #print("%d %d %f"%(v,vx,dis_f(vdata[v], vdata[vx])))
        dis = dis_f(vlist[v], vlist[vx])
        if(dis > DMAX) :
            mlist.append((v,vx))
            
        '''
        ds = int(dis*10)
        if ds in clist:
            clist[ds].append(vx)
            clist[ds].append(v)
        else:
            clist[ds] = []
        '''
    
cmap = {} 
for x,y in mlist:
    if x in cmap:
        #print(cmap[y])
        cmap[y] = cmap[x]
    elif y in cmap:
        #print(cmap[y])
        cmap[x] = cmap[y]
    else:
        cmap[x] = numC
        cmap[y] = numC
    


#for w in clist:
#    print("%d %d"%(w,len(set(clist[w]))))
