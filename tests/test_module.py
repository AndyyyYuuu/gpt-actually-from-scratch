import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.functional import softmax
from gpt_from_scratch.module import Linear, Sequential
from gpt_from_scratch.tensor import tensor
from gpt_from_scratch.saving import save, load
from gpt_from_scratch.parameter import Parameter

x = tensor([[1, 2], [3, 4], [5, 6], [7,8]])
net1 = Linear(2, 3)
net1.weight = Parameter(tensor([[1, 2, 3], [3, 2, 1]]))
net2 = Sequential([
    Linear(2, 3), 
    Linear(3, 2)
])
net3 = Linear(2, 3)
#print(net(x))
#print(softmax(model(x), dim=0))
print(net1.state_dict())
save(net1.state_dict(), "s.pt")
net3.load_state_dict(load("s.pt"))
print(net3.state_dict())