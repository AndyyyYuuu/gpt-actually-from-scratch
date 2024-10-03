
from . import tensor
def linear(input: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor=None): 
    return input @ weight + bias
