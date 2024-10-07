import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.module import Linear
from gpt_from_scratch.tensor import tensor

model = Linear(2, 3)

print(model(tensor([[1, 2], [2, 3]])))
