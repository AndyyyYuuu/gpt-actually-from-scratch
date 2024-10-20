'''
Note: 

The test cases in the following file were generated using an AI large language 
model (GPT-4o mini via chatgpt.com) for convenience and accuracy. While AI was
used to assist in generating these test cases, the implementation and
validation of the tests are the responsibility of the author.
'''

import unittest, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from gpt_from_scratch.tensor import Tensor, tensor, zeros, cat, sum
from gpt_from_scratch.utils import _enforce_type

class TestTensorSummation(unittest.TestCase):
    def test_1d_tensor_sum(self):
        t = tensor([1, 2, 3, 4, 5])
        self.assertEqual(sum(t), tensor(15))
        self.assertEqual(sum(t, dim=0), tensor(15))

    def test_2d_tensor_sum(self):
        t = tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(sum(t), tensor(21))
        self.assertEqual(sum(t, dim=0), tensor([5, 7, 9]))
        self.assertEqual(sum(t, dim=1), tensor([6, 15]))

    def test_3d_tensor_sum(self):
        t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(sum(t), tensor(36))
        self.assertEqual(sum(t, dim=0), tensor([[6, 8], [10, 12]]))
        self.assertEqual(sum(t, dim=1), tensor([[4, 6], [12, 14]]))
        self.assertEqual(sum(t, dim=2), tensor([[3, 7], [11, 15]]))

    def test_4d_tensor_sum(self):
        t = tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
        self.assertEqual(sum(t), tensor(136))
        self.assertEqual(sum(t, dim=0), 
                         tensor([[[10, 12], [14, 16]], [[18, 20], [22, 24]]]))
        self.assertEqual(sum(t, dim=1), 
                         tensor([[[6, 8], [10, 12]], [[22, 24], [26, 28]]]))
        self.assertEqual(sum(t, dim=2), 
                         tensor([[[4, 6], [12, 14]], [[20, 22], [28, 30]]]))
        self.assertEqual(sum(t, dim=3), 
                         tensor([[[3, 7], [11, 15]], [[19, 23], [27, 31]]]))

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


#print(sum(tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]), 2, True))
#print(sum(tensor([[1, 2], [3, 4]]),dim=1))
#print(cat((tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), 1))
#print(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
#unittest.main()
if __name__ == '__main__':
    unittest.main()