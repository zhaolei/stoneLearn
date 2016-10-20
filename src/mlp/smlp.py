# -*- coding: utf-8 -*-
'''

TODO
1. 输入检查
2. 自动初始化
3. batch 批量梯度 
4. 自定义激活函数
5. 支持 输出层softmax 包括softmax误差
6. 单层自定义
7. 增加单层偏置 b
'''

import numpy as np
from numpy import genfromtxt
import mstone

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
            return 1 - np.pow(x,2)
        return np.tanh(x)


    def sigmod(self, x=None, y=None, lerr=None, deriv=False):

        if(deriv==True):
            dx = x * (1 - x)
            dx = lerr * dx
            dloss = dx.dot(y.T)
            return dx, dloss 
 
        return 1/(1+np.exp(-x))

    def softmax(self, x, y=None, lerr=None, deriv=False):

        if deriv == True:
            lerr = lerr.reshape(lerr.shape[0])
            x[range(x.shape[0]), lerr.reshape(lerr.shape[0])] -= 1
            #dx = x.dot(y.T)
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
    

    # example (1,8)
    hideLayers = []
    
    # example (8,3),(3,5),(5,9)
    inputLayer = []

    # example (9,1)
    # no softmax
    outputLayer = []
    
    # rand
    paramLayers = []
    
    # layerdata
    levelLayers = []

    deltaLayers = []
    
    actFunctions = [] 
    actDxFunctions = [] 

    def __init__(self, inX=1, inY=1, outY=1, batch=20):
        self.inX = inX
        self.inY = inY
        self.outY = outY
        self.inputLayer.append((inX,inY))
        self.actObj = ActFun()
        
    def addLayer(self, intX=1, intY=1, act='sigmod'):
        self.hideLayers.append((intX,intY))
        #self.actFunctions.append(self.nonlinx)
        self.actFunctions.append(self.actObj.func(act))
        #self.actDxFunctions.append(self.actObj.dx_func('dx_%s'%act))

    def setOut(self,intOut):
        self.outputLayer.append((intOut))
 

    def initLayer(self):

        for shx in self.hideLayers :
            #syns = 2*np.random.random((shx[0],shx[1])) - 1
            syns = np.random.random((shx[0],shx[1]))/np.sqrt(shx[0]) 
            self.paramLayers.append(syns)

    def nonlin(self,x,deriv=False):

        if(deriv==True):
            return x*(1-x)
 
        return 1/(1+np.exp(-x))


    
    def printParam(self):
        print('input:')
        print self.inputLayer
        print('hide')
        print self.hideLayers
        #print('param')
        #print self.paramLayers
        print('func')
        print(self.actFunctions)

    def mtrain(self, tdata, tlab):
        for i in range(self.epochs):
            j = 0
            while j+self.batch < len(tdata): 
                #self.train(tdata, tlab)
                # 数据分片
                tmpX = tdata[j:j+self.batch]
                tmpY = tlab[j:j+self.batch]

                # 前向计算误差
                self.train_forward(tmpX, tmpY)
                #self.get_softmax_err(tmpX,tmpY, j)
                if j==0 :
                    #self.get_err(tmpX, tmpY, i)
                    #self.get_softmax_err(tmpX,tmpY, i)
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
        for (shx, actF) in zip(self.paramLayers, self.actFunctions):
            #lx = self.nonlin(np.dot(trainX, shx), deriv=False)
            lx = actF(np.dot(trainX, shx), deriv=False)
            #lx = self.sigmod(np.dot(trainX, shx), deriv=False)
            self.levelLayers.append(lx)
            trainX = lx
        return lx

    def train_outlayer(self, tdata, tlab, x):
        return x

    def train_hidelayer(self, tdata, tlab, x):
        return x

    '''梯度计算'''
    def train_delta(self, tdata, tlab, x): 

        self.deltaLayers = []
        #lx_err = tlab - self.levelLayers[-1]
        lx_err = tlab 
        for (shx,shxs,actF) in zip(self.levelLayers[::-1], self.paramLayers[::-1], self.actFunctions[::-1]):
            #lx_delta,lx_err = lx_err*actF(x=shx, y=shxs)
            #lx_delta = lx_err*self.nonlin(shx, deriv=True)
            #lx_delta, lx_err= self.dx_sigmod(shx, shxs, lx_err)
            #lx_delta, lx_err= actF(shx, shxs, lx_err, deriv=True)
            
            lx_delta, lx_err= actF(shx, shxs, lx_err, deriv=True)
            '''
            print 'xxxx %d' % x
            print lx_delta
            print lx_err
            exit()
            if x > 0:
                exit()
            if lx_delta[0][0] > 0 or (lx_delta[0][0] * -1) > 0:
                print ''
            else:
                print '00000:0000 : %d'%x
                #print lx_delta[0]
            '''
            #lx_err = lx_delta.dot(shxs.T)
            self.deltaLayers.append(lx_delta)

    

    
    '''参数更新'''
    def train_param(self,tdata, tlab):
        lx = self.levelLayers.pop()     
        
        for lx_delta in self.deltaLayers:
            
            lp = self.paramLayers.pop()     
            lx = self.levelLayers.pop()     
            
            ll = lx.T.dot(lx_delta)
            lp += -1 * lx.T.dot(lx_delta)
            self.paramLayers.insert(0,lp)
    
    '''计算误差'''
    def get_err(self, tdata, tlab, i):
        lx_err = tlab - self.levelLayers[-1]
        print "Error batch[%d] lost: %f"%(i , np.mean(np.abs(lx_err)))

    def get_softmax_err(self, tdata, tlab, i):
        probs = self.levelLayers[-1]
        
        tp = probs[range(probs.shape[0]), tlab.reshape(tlab.shape[0])]
        corect_logprobs = -np.log(tp)
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        #data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        tloss = data_loss / tlab.shape[0] 

        print "Error batch[%d] lost: %f"%(i , tloss)

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
        return tloss
        
    def calculate(self, tdata, tlab, i):
        plab = self.predict(tdata)
        lost = self.get_softmax_lost(plab, tlab) 
            
        print "Error batch[%d] lost: %f"%(i , lost)
        

    def train(self, tdata, tlab):
        
        trainX = tdata
        self.levelLayers = []
        self.levelLayers.append(trainX)
        for shx in self.paramLayers:
            lx = self.nonlin(np.dot(trainX, shx), deriv=False)
            self.levelLayers.append(lx)
            trainX = lx


        lx_err = tlab - self.levelLayers[-1]
        print "Error:[0]" + str(np.mean(np.abs(lx_err)))

        #lx_err = None
        self.deltaLayers = []
        for (shx,shxs) in zip(self.levelLayers[::-1], self.paramLayers[::-1]):
            lx_delta = lx_err*self.nonlin(shx, deriv=True)
            lx_err = lx_delta.dot(shxs.T)
            self.deltaLayers.append(lx_delta)

            '''
            if lx_err is None:
                #lx_err = tlab - shx 
                #print "Error:[0]" + str(np.mean(np.abs(lx_err)))
                #print lx_err
                lx_delta = lx_err*self.nonlin(shx, deriv=True)
                self.deltaLayers.append(lx_delta)
                lx_err = lx_delta.dot(shxs.T)
                #lx_delta = lx_err*self.nonlin(shx, deriv=True)
            else:

                lx_delta = lx_err*self.nonlin(shx, deriv=True)
                self.deltaLayers.append(lx_delta)
                lx_err = lx_delta.dot(shxs.T)
            '''

        
        #for((shx,shxs) in (self.deltaLayers[::-1], self.paramLayers[::-1])):
        #print self.paramLayers
        #for ikey in range(len(self.paramLayers)):
        lx = self.levelLayers.pop()     
        
        for lx_delta in self.deltaLayers:
            
            lp = self.paramLayers.pop()     
            lx = self.levelLayers.pop()     
            
            lp += lx.T.dot(lx_delta)
            self.paramLayers.insert(0,lp)

        
            
if __name__ == '__main__':
    print 'test'
    obx = Smlp(10,3)
    obx.setOut(1)
    obx.addLayer(2,5)
    obx.addLayer(5,7)
    obx.addLayer(7,1)
    obx.initLayer()
    obx.printParam()

    '''
    X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
 
    y = np.array([[0],
            [1],
            [1],
            [0]])
    '''
    dataX = genfromtxt('%s/smlp_data_x.csv'%mstone.data_path, delimiter=',')
    dataY = genfromtxt('%s/smlp_data_y.csv'%mstone.data_path, delimiter=',')
    dataY = dataY.reshape(dataY.shape[0],1)

    obx.mtrain(dataX,dataY)
                
