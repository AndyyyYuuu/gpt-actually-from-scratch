from typing import Union
from .tensor import Tensor
from .utils import _enforce_type

class Parameter(Tensor): 
    def __init__(self, value: Tensor): 
        
        _enforce_type(value, Tensor)
        super().__init__(value.data, value.shape, value.stride)

    def __repr__(self): 
        return f"Parameter(shape={self.shape}, data={self.tolist()})"
    
    def copy(self, other): 
        self.data = other.data.copy()
        self.shape = other.shape.clone()
        self.stride = other.stride.copy()
    
