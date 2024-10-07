from . import functional as F
from . import tensor
from .parameter import Parameter

class Module: 

    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs)
    def forward(self): 
        raise NotImplementedError("method `forward` not implemented for subclass of Module")


class Linear(Module): 
    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tensor.ones(in_features, out_features))
        self.bias = Parameter(tensor.ones(out_features))
    def forward(self, x): 
        return F.linear(x, self.weight, self.bias)
