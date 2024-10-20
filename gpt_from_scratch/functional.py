
from . import tensor
from .utils import _enforce_type
import math

def linear(input: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor=None):
    _enforce_type(input, tensor.Tensor)
    _enforce_type(weight, tensor.Tensor)
    _enforce_type(bias, tensor.Tensor)
    return input @ weight + bias


def softmax(input: tensor.Tensor, dim: int=None): 
    _enforce_type(input, tensor.Tensor)
    if not dim is None: _enforce_type(dim, int)
    return math.e**input / tensor.sum(math.e**input, dim=dim, keepdim=True)


