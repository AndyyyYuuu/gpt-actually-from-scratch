import functional as F
import tensor
from parameter import Parameter

class Module: 

    def __call__(self): 
        self.forward()
    def forward(self): 
        raise NotImplementedError("method `forward` not implemented for subclass of Module")

class Linear(Module): 
    def __init__(self, in_features: int, out_features: int): 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tensor.zeros(in_features, out_features))
        self.bias = Parameter(tensor.zeros(out_features))
