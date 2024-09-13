import unittest
from gpt_from_scratch.tensor import *


class TestTensorMatMul(unittest.TestCase):
    def test_2x2_matmul(self):
        a = tensor([[1, 2], [3, 4]])
        b = tensor([[5, 6], [7, 8]])
        result = a @ b
        expected = tensor([[19, 22], [43, 50]])
        self.assertEqual(result, expected)

    def test_2x3_3x2_matmul(self):
        a = tensor([[1, 2, 3], [4, 5, 6]])
        b = tensor([[7, 8], [9, 10], [11, 12]])
        result = a @ b
        expected = tensor([[58, 64], [139, 154]])
        self.assertEqual(result, expected)

    def test_1x3_3x1_matmul(self):
        a = tensor([[1, 2, 3]])
        b = tensor([[4], [5], [6]])
        result = a @ b
        expected = tensor([[32]])
        self.assertEqual(result, expected)

    def test_incompatible_shapes(self):
        a = tensor([[1, 2], [3, 4]])
        b = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError):
            _ = a @ b

print(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).stride)
print(cat((tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), 1))
#print(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
#unittest.main()
