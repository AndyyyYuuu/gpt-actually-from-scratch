from typing import Self, Union


class Tensor:

    def __init__(self, tensor: Self): 
        self.size = tensor.size
        self.data = tensor.data
        self.stride = tensor.stride
        

    def __init__(self, data: list) -> None:
        self.size = self._detect_shape(data)
        self.data = _flatten_list(data)
        self.stride = self._get_stride()
    
    def __getitem__(self, index: list) -> float: 
        flat_index = sum([i*j for i, j in zip(self.stride, index)])
        return self.data[flat_index]

    def __setitem__(self, index: list, value) -> None: 
        flat_index = sum([i*j for i, j in zip(self.stride, index)])
        self.data[flat_index] = value

    
    def _get_stride(self): 
        stride = [1]*self.size.dim()
        for i in range(self.size.dim()-1, 0, -1): 
            stride[i-1] = self.size[i]*stride[i]
        return stride
    
    def __eq__(self, other): 
        return other.size == self.size and all([i==j for i, j in zip(self.data, other.data)])

    def __add__
    
    def _detect_shape(self, data: list) -> tuple: 
        shape = []
        depth_data = data
        while isinstance(depth_data, (list, tuple)): 
            shape.append(len(depth_data))
            if len(depth_data) == 0: 
                break
            depth_data = depth_data[0]
        return Size(*shape)

    def shape(self) -> tuple: 
        return self.size
    
    def __repr__(self) -> str: 
        return f"Tensor(shape={self.shape()}, data={self.tolist()})"
        
    
    @staticmethod
    def _data_eq(data1: list, data2: list) -> bool: 
        
        if len(data1) != len(data2): 
            return False
        if isinstance(data1[0], (list, tuple)): 
            return all([Tensor._data_eq(data1[i], data2[i]) for i in range(len(data1))])
        else: 
            return all([data1[i] == data2[i] for i in range(len(data1))])
    
    def __matmul__(self, other: Self) -> Self:
        
        if self.size[1] != other.size[0]: 
            if other.size[1] == self.size[0]: 
                self, other = other, self
            else: 
                raise ValueError("The number of columns in the first matrix must equal the number of rows in the second")
        n = self.size[0]
        m = other.size[0]
        p = other.size[1]
        c = zeros(n, p)
        for i in range(n): 
            for j in range(p): 
                for k in range(m): 
                    c[i,j] += self[i,k]*other[k,j]
                    
        return c
    
    def tolist(self) -> Union[list, float]: 
        def build_list(dim, index): 
            if dim == self.size.dim(): 
                return self[index]
            return [build_list(dim + 1, index + [i]) for i in range(self.size[dim])]
        return build_list(0, [])
    
    def transpose(self, dim1: int, dim2: int) -> None: 
        self.size[dim1], self.size[dim2] = self.size[dim2], self.size[dim1]
        self.stride[dim1], self.stride[dim2] = self.stride[dim2], self.stride[dim1]
        return self


def _flatten_list(data: list): 
    if isinstance(data, list) and len(data) != 0 and isinstance(data[0], list): 
        data = sum([_flatten_list(i) for i in data], start=[])
    return data

def flatten(self, tensor: Tensor) -> Tensor: 
    flat_tensor = Tensor(tensor)
    prod = 1
    for i in self.stride: 
        prod*=i
    flat_tensor.shape = (prod,)
    flat_tensor.stride = (1,)
    return flat_tensor

def _num_list(shape: Union[tuple, list], num: int) -> list: 
    if len(shape) == 1: 
        return shape[0] * [num]
    return shape[0] * [_num_list(shape[1:], num)]

def zeros(*shape: Union[tuple, list]) -> Tensor: 
    return Tensor(_num_list(shape, num=0))

def ones(*shape: Union[tuple, list]) -> Tensor:
    return Tensor(_num_list(shape, num=1))


class Size: 
    def __init__(self, *sizes: list) -> None: 
        self.data = list(sizes)
    
    def __eq__(self, other: Union[Self, int]) -> bool: 

        if self.dim() != other.dim(): 
            return False
        return all([i == j for i, j in zip(self, other)])
    
    def __ne__(self, other: Self) -> bool: 
        return not self == other
    
    def __getitem__(self, dim: int) -> int: 
        return self.data[dim]
    
    def __setitem__(self, dim: int, value: int) -> int: 
        self.data[dim] = value
    
    def __repr__(self): 
        return f"Size({str(self.data)})"
    
    def total(self): 
        prod = 1
        for i in self.data: 
            prod *= i
        return prod
    
    def dim(self): 
        return len(self.data)