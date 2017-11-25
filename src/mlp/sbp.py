# -*- coding: utf-8 -*-
'''
单行双层神经网络demo
'''
import numpy as np
from numpy import genfromtxt


#X = np.linspace(1,100,100)
#X = X.reshape(100,1)

X = np.array([[1]]) 
Y = np.array([0.])


'''第一层5个神经元'''
W11 = np.random.random([1,5])
b11 = np.random.random([1,5])

'''输出层1个神经元 因为输出是1纬'''
W21 = np.random.random([5,1])
b21 = np.random.random([1,1])

for n in range(1000):
    #print('-------train---------')

    # 第一层前向计算
    dn0 = X * W11 + b11
    #print('%s dot %s => %s'%(str(X.shape), str(W11.shape), str(dn0.shape)))
    # 第一层前向计算结果激活
    adn0 = 1/(1+np.exp(-dn0))

    # 第一层前向计算激活导数
    dt0 = adn0*(1-adn0)
    

    # 第二层前向计算
    dn1 = dn0.dot(W21) + b21
    #print('%s dot %s => %s'%(str(dn0.shape), str(W21.shape), str(dn1.shape)))
    # 第二层前向计算结果激活
    adn1 = 1/(1+np.exp(-dn1))

    # 第二层前向计算结果激活导数
    dt1 = adn1*(1-adn1)

    #losx = Y*np.log(adn1) + (1-Y)*np.log(1-adn1)

    # 输出层误差
    losx = np.power(Y - adn1,2)

    # 输出误差函数导数
    dt = Y - adn1 

    delta = dt * dt1
    da = adn0.T.dot(delta)

    #反向梯度梯度下降
    W21 = W21 + da * 0.1 
    b21 = b21 + delta * 0.1 

    error0 = delta.dot(W21.T)
    delta0 = error0 * dt0

    da0 = X.T.dot(delta0)

    #反向梯度梯度下降
    W11 = W11 + da0 * 0.1 
    b11 = b11 + delta0 * 0.1 

    
    #print((Y-adn1)[0:10])
    if n % 100 == 0 :    
        print("lost %f y^: %f y: %f"%(losx[[0]], adn1[[0]], Y[[0]]))

