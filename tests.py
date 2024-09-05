import unittest
from gpt_from_scratch.ops import Tensor


class TestTensorMatMul(unittest.TestCase):
    def test_2x2_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a @ b
        expected = Tensor([[19, 22], [43, 50]])
        self.assertEqual(result, expected)

    def test_2x3_3x2_matmul(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([[7, 8], [9, 10], [11, 12]])
        result = a @ b
        expected = Tensor([[58, 64], [139, 154]])
        self.assertEqual(result, expected)

    def test_1x3_3x1_matmul(self):
        a = Tensor([[1, 2, 3]])
        b = Tensor([[4], [5], [6]])
        result = a @ b
        expected = Tensor([[32]])
        self.assertEqual(result, expected)

    def test_incompatible_shapes(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError):
            _ = a @ b


if __name__ == '__main__':
    unittest.main()