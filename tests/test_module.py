import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.functional import softmax
from gpt_from_scratch.module import Linear, Sequential, Module
from gpt_from_scratch.tensor import tensor
from gpt_from_scratch.saving import save, load
from gpt_from_scratch.parameter import Parameter

x = tensor([[1, 2], [3, 4], [6, 7], [9, 10]])

#print(net(x))
#print(softmax(model(x), dim=0))
class FeedForward(Module): 
    def __init__(self, input_features, hidden_features, output_features): 
        super().__init__()
        self.feedforward = Sequential(
            Linear(input_features, hidden_features),
            Linear(hidden_features, output_features)
        )
    def forward(self, x): 
        return self.feedforward(x)

net1 = FeedForward(2, 3, 2)
net2 = Linear(2, 3)
print(net2(x))

print(net1(x))

loaded = load("save.pt")
print(loaded)
net1.load_state_dict(loaded)

print("\n2:", net1.feedforward.modules[0].weight)
print(net1(x))