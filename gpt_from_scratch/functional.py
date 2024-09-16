
import tensor
def linear(input: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor): 
    return sum(input * weight) + bias