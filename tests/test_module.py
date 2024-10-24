import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.functional import softmax
from gpt_from_scratch.module import Linear, Sequential
from gpt_from_scratch.tensor import tensor

x = tensor([[1, 2], [3, 4], [5, 6], [7,8]])
net = Sequential([
    Linear(2, 3), 
    Linear(3, 2)
])
#print(net(x))
#print(softmax(model(x), dim=0))
print(net.parameters())