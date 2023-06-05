[中文](./README_zh.md) | [English](#)
# handwriting_deep_learning_framework

reference：https://blog.csdn.net/qq_43790749/article/details/112130630

## main program
[general_neural_network_framework.py](./general_neural_network_framework.py)


## Computational graph creation 
In the initialization of the base class Node, self.inputs and self.outputs record the relationship between nodes, so as to establish the calculation graph.


```
class Node:
    """
    We use this node class as the basic module of this neural network
    """
    def __init__(self,inputs=[],name=None,is_trainable=False):
        
        self.inputs = inputs  # The input of this node, the input is a list of nodes
        self.outputs = []     # output node of this node
        self.name = name
        self.is_trainable = is_trainable
        for n in self.inputs:
            # This self node happens to be the output node of this input node (n), 
            # thus establishing a connection relationship
            n.outputs.append(self)  
            
        self.value = None  # Each node must have a corresponding value
        self.gradients = {}  # The gradient of each node to the previous node

    def forward(self):
        """
        Reserve a non-implemented interface first, implement it in its subclasses, 
        and require its subclasses to be implemented. If it is not implemented, 
        an error will be reported.
        """
        raise NotImplemented  
        
    def backward(self):
        
        raise NotImplemented
        
    def __repr__(self):
        
        return "Node:{}".format(self.name)  
```

## topological sort

The purpose of topological sorting is to obtain an ordered list of nodes。
When calculating the output value of a node in forward，Ensure that the input value of this node has been calculated，In other words, the parent node connected to this node has already been counted.
Also during backpropagation, according to the topologically sorted nodes, before calculate the gradient of a node , ensure that the child nodes connected to the node have already calculated the gradient.


## Forward calculation and Gradient Backpropagation
After the calculation graph is topologically sorted, the forward of each node is called for forward calculation according to the sorted order, and then the backward is called for gradient update during backpropagation.
In addition, multi-dimensional forward calculation and backpropagation are performed in the form of matrix operations.

### python package release
when you alread finish the handwriting_deep_learning_framework, you want to public you code, you can release your python package to pypi
reference: https://blog.csdn.net/qq_43790749/article/details/112134520

