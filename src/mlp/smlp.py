# -*- coding: utf-8 -*-
'''

TODO
1. 输入检查
2. batch 批量梯度 
3. 正则化
'''

import numpy as np
from numpy import genfromtxt
import mstone
import math

class ActFun:
    
    def func(self, f):
        if f == 'sigmod':
            return self.sigmod
        elif f == 'softmax':
            return self.softmax
        elif f == 'tanh':
            return self.tanh

        return self.tanh

    def tanh(self, x=None, y=None, lerr=None, deriv=False):
        if(deriv==True):
            dx = 1 - pow(x,2)
            dx = lerr * dx
            dloss = dx.dot(y.T)
            return dx,dloss
        return np.tanh(x)


    def sigmod(self, x=None, y=None, lerr=None, deriv=False):

        if(deriv==True):
            dx = x * (1 - x)
            dx = lerr * dx
            dloss = dx.dot(y.T)
            return dx, dloss 
 
        tmp = 1/(1+np.exp(-x))
        return tmp

    def softmax(self, x, y=None, lerr=None, deriv=False):

        if deriv == True:
            lerr = lerr.reshape(lerr.shape[0])
            x[range(x.shape[0]), lerr.reshape(lerr.shape[0])] -= 1
            dx = x

            dloss = dx.dot(y.T)
            return dx,dloss

        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return  probs
        #return np.divide(x.T, np.sum(x.T,axis=0),dtype=float).T
    

class Smlp:
    inX = 1
    inY = 1
    outY = 1
    batch = 20
    epochs = 5000
    alpa = 0.1
    

    # example (8,3),(3,5),(5,9)
    inputLayer = []

    # rand
    paramLayers = []
    
    # layerdata
    levelLayers = []

    deltaLayers = []
    
    setConfig = []

    def __init__(self, alpa=0.1, batch=20, epochs=5000):
        self.actObj = ActFun()
        self.batch = batch
        self.alpa = alpa
        self.epochs = epochs
        
    def addLayer(self, intY=1, act='sigmod'):
        conf = {}
        #conf['x'] = intX
        conf['y'] = intY
        conf['actFun'] = act 
        self.setConfig.append(conf)

    def setOut(self,intOut):
        self.outputLayer.append((intOut))
 

    '''参数初始化'''
    def initLayer(self, inX):

        #for shx in self.hideLayers :
        initX = inX 
        for val in self.setConfig :
            syns = {}
            #syns['W'] = np.random.random((val['x'],val['y']))/np.sqrt(val['x']) 
            syns['W'] = np.random.random((initX,val['y']))/np.sqrt(initX) 
            syns['b'] = np.zeros((1,val['y'])) 
            syns['act'] =  self.actObj.func(val['actFun'])
            #syns = np.random.random((shx[0],shx[1]))/np.sqrt(shx[0]) 
            self.paramLayers.append(syns)
            initX = val['y']

    def mtrain(self, tdata, tlab):
        self.initLayer(tdata.shape[1])

        for i in range(self.epochs):
            j = 0
            while j+self.batch < len(tdata): 
                #self.train(tdata, tlab)
                # 数据分片
                tmpX = tdata[j:j+self.batch]
                tmpY = tlab[j:j+self.batch]

                #print '************ %d'%j
                # 前向计算误差
                self.train_forward(tmpX, tmpY)
                if j==0 :
                    self.calculate(tmpX,tmpY, i)

                # 误差反向 计算
                self.train_delta(tmpX, tmpY, j)
                
                # 更新参数
                self.train_param(tmpX, tmpY)
                j += self.batch



    '''前向计算'''
    def train_forward(self, tdata, tlab=None):
        trainX = tdata
        self.levelLayers = []
        self.levelLayers.append(trainX)
        for shx in self.paramLayers:
            lx = shx['act'](np.dot(trainX, shx['W']) + shx['b'], deriv=False)
            self.levelLayers.append(lx)
            trainX = lx

        return trainX 


    '''梯度计算'''
    def train_delta(self, tdata, tlab, x): 

        self.deltaLayers = []
        #lx_err = tlab - self.levelLayers[-1]
        lx_err = tlab 
        for (shx,shxs) in zip(self.levelLayers[::-1], self.paramLayers[::-1]):
            #lx_delta, lx_err= actF(shx, shxs['W'], lx_err, deriv=True)
            lx_delta, lx_err= shxs['act'](shx, shxs['W'], lx_err, deriv=True)
            self.deltaLayers.append(lx_delta)

    
    '''参数更新'''
    def train_param(self,tdata, tlab):
        lx = self.levelLayers.pop()     
        
        for lx_delta in self.deltaLayers:
            
            lp = self.paramLayers.pop()     
            lx = self.levelLayers.pop()     
            
            ll = lx.T.dot(lx_delta)
            lp['W'] += -self.alpa * lx.T.dot(lx_delta)
            lp['b'] += -self.alpa * np.sum(lx_delta, axis=0)
            
            self.paramLayers.insert(0,lp)
    
    '''计算误差'''
    def get_err(self, tdata, tlab, i):
        lx_err = tlab - self.levelLayers[-1]
        print "Error batch[%d] lost: %f"%(i , np.mean(np.abs(lx_err)))

    def predict(self, tdata):
        return self.train_forward(tdata)

    def get_softmax_lost(self, plab, tlab):
        probs = plab
        tp = probs[range(probs.shape[0]), tlab.reshape(tlab.shape[0])]
        corect_logprobs = -np.log(tp)
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        #data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        tloss = data_loss / tlab.shape[0] 
        return tloss,tp.mean()
        
    def calculate(self, tdata, tlab, i):
        plab = self.predict(tdata)
        lost,tmean = self.get_softmax_lost(plab, tlab) 
            
        print "Error batch[%d] lost: %f mean: %f"%(i , lost, tmean)
        
