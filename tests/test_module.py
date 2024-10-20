import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.functional import softmax
from gpt_from_scratch.module import Linear
from gpt_from_scratch.tensor import tensor

model = Linear(2, 3)
x = tensor([[1, 2], [3, 4], [5, 6], [7,8]])
print(model(x))
print(softmax(model(x), dim=0))
#print(model.parameters())