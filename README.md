# handwriting_deep_learning_framework

参考：https://blog.csdn.net/qq_43790749/article/details/112130630

## 主要程序
通用神经网络框架.py


## 计算图的建立 
在基类Node中 初始化中 self.inputs 与self.outputs 记录了节点之间的关系，以此将计算图建立好。


```
class Node:
    """
    我们把这个Node类作为这个神经网络的基础模块
    """
    def __init__(self,inputs=[],name=None,is_trainable=False):
        
        self.inputs = inputs  #这个节点的输入，输入的是Node组成的列表
        self.outputs = []     #这个节点的输出节点
        self.name = name
        self.is_trainable = is_trainable
        for n in self.inputs:
            n.outputs.append(self)  #这个节点正好对应了这个输人的输出节点，从而建立了连接关系
            
        self.value = None  #每个节点必定对应有一个值
        self.gradients = {}  #每个节点对上个节点的梯度，

    def forward(self):
        """
        先预留一个方法接口不实现，在其子类中实现,且要求其子类一定要实现，不实现的时话会报错。
        """
        raise NotImplemented  
        
    def backward(self):
        
        raise NotImplemented
        
    def __repr__(self):
        
        return "Node:{}".format(self.name)  
```

## 拓扑排序

拓扑排序的目的是，获得一个排好顺序的节点列表.
前向计算一个节点的输出值时，保证这个节点的输入值已经计算好了，换句话说，与这个节点相连的父节点已经算好值了。
同样反向传播时，将拓扑排序好的列表进行倒序计算，在计算一个节点的梯度，保证该节点的子节点已经算好梯度了。


## 前向计算与梯度反向传播
将计算图进行拓扑排序后，按照排序后的顺序调用每个节点的forward进行前向计算，然后在反向传播的时候调用backward 进行梯度更新。
多维度的前向计算与反向传播 是采用矩阵运算的形式进行运算的。

### 包的发布
https://blog.csdn.net/qq_43790749/article/details/112134520

