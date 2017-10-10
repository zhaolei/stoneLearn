import numpy as np


X = np.linspace(1,100,100)
X = X.reshape(100,1)

#Y = np.sin(X)
Y = X*0.5 + 0.5
print(Y.shape)
print(X.shape)

W11 = np.random.random([1,5])
b11 = np.random.random([5])
#W11 = np.zeros([1,2])
#b11 = np.zeros([2])

W21 = np.random.random([5,1])
b21 = np.random.random([1])

for n in range(500):
    dn0 = X.dot(W11) + b11
    adn0 = 1/(1+np.exp(-dn0))

    dt = adn0*(1-adn0)
    

    dn1 = dn0.dot(W21) + b21
    adn1 = 1/(1+np.exp(-dn1))

    #losx = Y*np.log(adn1) + (1-Y)*np.log(1-adn1)
    print(adn1.tolist().count(1.))
    losx = (1-Y)*np.log(1-adn1) 


    errs = np.average(Y-adn1)
    W21 = W21 -(W21*(errs)*0.001)
    b21 = b21 -(b21*(errs)*0.001)

    W11 = W11 - (W11*(errs)*np.average(dt)*0.001)
    b11 = b11 - (b11*(errs)*np.average(dt)*0.001)
    
    #print((Y-adn1)[0:10])
    print("lost %f"%np.average(losx))

'''
print('--------p---------')
print(W11)
print(b11)
print(W21)
print(b21)
'''
