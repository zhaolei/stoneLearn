# -*- coding: utf-8 -*-
'''

TODO
1. 输入检查
2. 自动初始化
3. batch 批量梯度 
'''

import numpy as np
from numpy import genfromtxt
import mstone

class Smlp:
    inX = 1
    inY = 1
    outY = 1
    batch = 20
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

    def __init__(self, inX=1, inY=1, outY=1):
        self.inX = inX
        self.inY = inY
        self.outY = outY
        self.inputLayer.append((inX,inY))

    def addLayer(self, intX=1, intY=1):
        self.hideLayers.append((intX,intY))

    def setOut(self,intOut):
        self.outputLayer.append((intOut))

    def initLayer(self):

        for shx in self.hideLayers :
            syns = 2*np.random.random((shx[0],shx[1])) - 1
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
        print('param')
        print self.paramLayers

    def mtrain(self, tdata, tlab):
        for i in range(1200):
            j = 0
            while j+self.batch < len(tdata): 
                #self.train(tdata, tlab)
                tmpX = tdata[j:j+self.batch]
                tmpY = tlab[j:j+self.batch]

                self.train_forward(tmpX, tmpY)
                if i % 30 == 0:
                    self.get_err(tmpX, tmpY)
                self.train_delta(tmpX, tmpY)
                self.train_param(tmpX, tmpY)
                j += self.batch
            

    '''前向计算'''
    def train_forward(self, tdata, tlab):
        trainX = tdata
        self.levelLayers = []
        self.levelLayers.append(trainX)
        for shx in self.paramLayers:
            lx = self.nonlin(np.dot(trainX, shx), deriv=False)
            self.levelLayers.append(lx)
            trainX = lx

    '''梯度计算'''
    def train_delta(self, tdata, tlab): 

        self.deltaLayers = []
        lx_err = tlab - self.levelLayers[-1]
        for (shx,shxs) in zip(self.levelLayers[::-1], self.paramLayers[::-1]):
            lx_delta = lx_err*self.nonlin(shx, deriv=True)
            lx_err = lx_delta.dot(shxs.T)
            self.deltaLayers.append(lx_delta)
    
    '''参数更新'''
    def train_param(self,tdata, tlab):
        lx = self.levelLayers.pop()     
        
        for lx_delta in self.deltaLayers:
            
            lp = self.paramLayers.pop()     
            lx = self.levelLayers.pop()     
            
            lp += lx.T.dot(lx_delta)
            self.paramLayers.insert(0,lp)
    
    '''计算误差'''
    def get_err(self, tdata, tlab):
        lx_err = tlab - self.levelLayers[-1]
        print "Error:[0]" + str(np.mean(np.abs(lx_err)))
        

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
    #obx.printParam()

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
                
