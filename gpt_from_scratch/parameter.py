from typing import Union
from .tensor import Tensor
from .utils import _enforce_type

class Parameter(Tensor): 
    def __init__(self, value: Tensor): 
        
        _enforce_type(value, Tensor)
        super().__init__(value.data, value.size, value.stride)
