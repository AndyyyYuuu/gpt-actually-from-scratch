from typing import Union
from tensor import Tensor

class Parameter: 
    def __init__(self, value: Union[float, Tensor]): 
        self.value = value