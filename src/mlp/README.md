
## 亲手 打磨 一个深度神经网络 

### 包括算法

* bgd 批量梯度
* 基本矩阵计算
* softmax
* tanh sigmod

`
import smlp
from numpy import genfromtxt
import mstone

if __name__ == '__main__':
    obx = smlp.Smlp(10,3)
    obx.addLayer(2,5)
    obx.addLayer(5, 2, 'softmax')
    obx.initLayer()

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
    dataY = genfromtxt('%s/smlp_data_y.csv'%mstone.data_path, delimiter=',', dtype=int)
    dataY = dataY.reshape(dataY.shape[0],1)

    obx.mtrain(dataX,dataY)
`
                
