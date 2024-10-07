from typing import Self, Union
from .utils import _enforce_type
import builtins

class Tensor:

    def __init__(self, data: list, size: 'Size', stride: tuple): 
        
        _enforce_type(data, list, float)
        _enforce_type(size, Size)
        _enforce_type(stride, tuple, float)
        self.size = size
        self.data = data
        self.stride = stride
    

    def __getitem__(self, index: Union[int, slice, list[Union[int, slice]]]) -> Union[float, list]: 
        if isinstance(index, (int, slice)): 
            index = (index,)
        
        _enforce_type(index, tuple)
        index = list(index)
        
        if all([isinstance(i, int) for i in index]) and len(index) == self.size.dim():
            flat_index = builtins.sum([i*j for i, j in zip(self.stride, list(index))])
            return self.data[flat_index]
        
        if len(index) > self.size.dim(): 
            raise IndexError(f"Tensor index out of range (expected {self.size.dim()} dimensions, found {len(index)})")
        while len(index) < self.size.dim(): 
            index.append(slice(None))
        
        for i in range(len(index)): 
            if isinstance(index[i], int): 
                index[i] = slice(index[i], index[i]+1, None)
            elif not isinstance(index[i], slice): 
                raise TypeError(f"Tensor index must be list of int or slice (found {type(index[i])})")
            
        result = zeros(*[len(range(self.size[i])[index[i]]) for i in range(len(index))])
        

        def _get_slice(slices: list[slice], pos_self: list[int], pos_result: list[int], depth=0) -> None: 
            if len(slices) == 0: 
                result[pos_result] = self[pos_self]
            else: 
                for i,j in enumerate(range(self.size[depth])[slices[0]]): 
                    _get_slice(slices[1:], pos_self+[j], pos_result+[i], depth+1)
        
        _get_slice(index, [], [])

        # Squeeze sliced dimensions
        new_size = []
        new_stride = []
        for i in range(self.size.dim()): 
            if self.size[i] == 1 or result.size[i] != 1:
                new_size.append(result.size[i])
                new_stride.append(result.stride[i])
        result.size = Size(new_size)
        result.stride = new_stride
        return result


    def __setitem__(self, index: list, value) -> None: 
        _enforce_type(index, tuple)
        flat_index = builtins.sum([i*j for i, j in zip(self.stride, index)])
        self.data[flat_index] = value

    def clone(self) -> Self: 
        return Tensor(self.data, self.size.clone(), self.stride.copy())

    def shape(self) -> 'Size': 
        return self.size
    
    def __repr__(self) -> str: 
        return f"Tensor(shape={self.shape()}, data={self.tolist()})"
    
    def __eq__(self, other: Self) -> bool: 
        _enforce_type(other, Tensor)
        return other.size == self.size and all([i==j for i, j in zip(self.data, other.data)])

    def __add__(self, other: Union[Self, float]) -> Self: 
        
        if isinstance(other, Tensor) and other.size == Size([]):
            other = other[0]
        if isinstance(other, (float, int)): 
            result = self.clone()
            for i in range(len(result.data)): 
                result.data[i] += other
            
            return result

        if self.size != other.size: 
            _self = self.clone()
            _enforce_type(other, Tensor)
            other = other.clone()
            while other.size.dim() < _self.size.dim(): 
                other.unsqueeze(0)
            while other.size.dim() > _self.size.dim(): 
                _self.unsqueeze(0)
            
            for i, j in zip(_self.size, other.size): 
                if i != j and i != 1 and j != 1:
                    raise ValueError(f"Incompatible addition for tensors of shapes {self.size} and {other.size}.")
            
            if _self.size.total() < other.size.total(): 
                _self.expand(other.size)
            else: 
                other.expand(self.size)
    
        result = zeros(*self.size)
        def _add_tensors(t1, t2, output, index=[]): 
            if len(index) == t1.size.dim(): 
                output[index] = t1[index] + t2[index]
            else: 
                for i in range(t1.size[len(index)]): 
                    _add_tensors(t1, t2, output, index + [i])
        _add_tensors(self, other, result)
        return result
            
    
    def __mul__(self, other: Union[Self, int, float]) -> Self: 
        if isinstance(other, (float, int)): 
            result = self.clone()
            for i in range(len(result.data)): 
                result[i] *= other
            return result 
        elif isinstance(other, self): 
            if other.size == self.size: 
                result = zeros(*self.size)
                def _mul_tensors(t1, t2, output, index=[]): 
                    if len(index) == t1.size.dim(): 
                        output[index] = t1[index] * t2[index]
                    else: 
                        for i in range(t1.size[len(index)]): 
                            _mul_tensors(t1, t2, output, index + [i])
                _mul_tensors(self, other, result)
                return result
            else: 
                raise ValueError(f"Tensors must have the same shape for element-wise multiplication. Found {other.size} and {self.size}.")
        else: 
            raise TypeError(f"Expected int, float, or Tensor for scalar or element-wise multiplication. Found {type(other)}.")
        
    
    @staticmethod
    def _data_eq(data1: list, data2: list) -> bool: 
        
        if len(data1) != len(data2): 
            return False
        if isinstance(data1[0], (list, tuple)): 
            return all([Tensor._data_eq(data1[i], data2[i]) for i in range(len(data1))])
        else: 
            return all([data1[i] == data2[i] for i in range(len(data1))])
    
    def __matmul__(self, other: Self) -> Self:
        _self = self.clone()
        if _self.size.dim() <= 1: 
            _self.unsqueeze(0)
        if other.size.dim() <= 1: 
            other.unsqueeze(0)
        if _self.size.dim() >= 2 and _self.size[1] != other.size[0]: 
            if other.size[1] == _self.size[0]: 
                _self, other = other, _self
            else: 
                raise ValueError(f"The number of columns in the first matrix must equal the number of rows in the second (found {_self.size} and {other.size})")
        n = _self.size[0]
        m = other.size[0]
        p = other.size[1]
        c = zeros(n, p)
        for i in range(n): 
            for j in range(p): 
                for k in range(m): 
                    c[i,j] += _self[i,k]*other[k,j]
        if c.size[0] == 1:
            c.squeeze(0)
        return c
    
    def tolist(self) -> Union[list, float]: 
        def build_list(dim, index): 
            if dim == self.size.dim(): 
                return self[index]
            return [build_list(dim + 1, index + [i]) for i in range(self.size[dim])]
        return build_list(0, [])

    def unsqueeze(self, dim: int) -> None: 
        _enforce_type(dim, int)
        self.size.data.insert(dim, 1)
        if dim != 0: 
            self.stride.insert(dim, self.stride[dim-1])
        else: 
            self.stride.insert(dim, 1)
    
    def squeeze(self, dim: int=None) -> None: 
        if dim is None: 
            new_size = []
            new_stride = []
            for i in range(self.size.dim()): 
                if self.size[i] != 1: 
                    new_size.append(self.size[i])
                    new_stride.append(self.stride[i])
            self.size = Size(new_size)
            self.stride = new_stride
        else: 
            _enforce_type(dim, int)
            if self.size[dim] == 1:
                self.stride.pop(dim)
                self.size.data.pop(dim)
            else: 
                raise ValueError(f"squeeze() expected a dimension of size 1 at index {dim}, got {self.size[dim]}")
            
    def expand(self, size: 'Size') -> None:
        _enforce_type(size, Size)
        for i in range(self.size.dim()): 
            if self.size[i] > size[i]:
                raise ValueError(f"Tensor of size {self.size} is too large to be expanded to {size}.")
            elif self.size[i] < size[i]: 
                if self.size[i] == 1:
                    self.size[i] = size[i]
                    self.stride[i] = 0
                else: 
                    raise ValueError(f"Tensor of size {self.size} cannot be expanded to {size}.")

    
def transpose(input: Tensor, dim1: int, dim2: int) -> Tensor: 
    _enforce_type(input, Tensor)
    _enforce_type(dim1, int)
    _enforce_type(dim2, int)
    output = input.clone()
    output.size[dim1], output.size[dim2] = input.size[dim2], input.size[dim1]
    output.stride[dim1], output.stride[dim2] = input.stride[dim2], input.stride[dim1]
    return output
    

def cat(tensors: tuple[Tensor, ...], dim: int=0): 
    _enforce_type(tensors, tuple, Tensor)
    _enforce_type(dim, int)
    for t in tensors[1:]: 
        for i in range(t.size.dim()): 
            if i != dim and t.size[i] != tensors[0].size[i]: 
                raise ValueError(f"Tensors of sizes {tensors[0].size, t.size} cannot be concatenated.")
    tensor_dims = tensors[0].size.dim()
    new_size = [tensors[0].size[i] if i != dim else builtins.sum([t.size[i] for t in tensors]) for i in range(tensor_dims)]
    result = zeros(*new_size)

    offset = 0  # Offset value for copying: sum of previous tensor lengths
    for i in range(len(tensors)): 
        def _copy_sublist(pos: list, depth: int): 
            if depth == 0: 
                result_pos = pos.copy()
                result_pos[dim] += offset
                result[result_pos] = tensors[i][pos]
            else: 
                for j in range(tensors[i].size[-depth]): 
                    _copy_sublist(pos+[j], depth-1)
        _copy_sublist([], tensor_dims)
        offset += tensors[i].size[dim]
    return result



def _get_stride(size: 'Size') -> list: 
    _enforce_type(size, Size)
    stride = [1]*size.dim()
    for i in range(size.dim()-1, 0, -1): 
        stride[i-1] = size[i]*stride[i]
    return stride
    
def _detect_shape(data: list) -> tuple: 
    shape = []
    depth_data = data
    while isinstance(depth_data, (list, tuple)): 
        shape.append(len(depth_data))
        if len(depth_data) == 0: 
            break
        depth_data = depth_data[0]
    return Size(shape)


def tensor(data: Union[list, float, int]) -> Tensor: 
    if isinstance(data, tuple): 
        data = list(data)
    elif isinstance(data, int): 
        data = data
        size = Size([])
    size = _detect_shape(data)
    data = _flatten_list(data)
    _enforce_type(data, list, float)
    stride = _get_stride(size)
    return Tensor(data, size, stride)

def _flatten_list(data: list): 
    if isinstance(data, list) and len(data) != 0 and isinstance(data[0], list): 
        data = builtins.sum([_flatten_list(i) for i in data], start=[])
    return data

def flatten(tensor: Tensor) -> Tensor: 
    _enforce_type(tensor, Tensor)
    flat_tensor = tensor.clone()
    prod = 1
    for i in tensor.size: 
        prod*=i
    flat_tensor.size = Size([prod])
    flat_tensor.stride = [1]
    return flat_tensor


def _num_tensor(size: 'Size', num: int) -> Tensor: 
    return Tensor(size.total()*[num], size, _get_stride(size))

def zeros(*shape: Union[tuple, list]) -> Tensor: 
    return _num_tensor(Size(shape), num=0)

def ones(*shape: Union[tuple, list]) -> Tensor:
    return _num_tensor(Size(shape), num=1)


def sum(input: Tensor, dim:int=None) -> Union[float, Tensor]: 
    _enforce_type(input, Tensor)
    
    if dim is None: 
        return tensor(builtins.sum(input.data))

    _enforce_type(dim, int)

    if dim >= input.size.dim(): 
        raise ValueError(f"dimension {dim} out of range for {input.size.dim()}-d Tensor")
    
    if input.size.dim() == 1: 
        return tensor(builtins.sum(input.data))
    
    output_tensor = zeros(*(input.size[:dim]+input.size[dim+1:]))
    total_slice = [slice(None)] * input.size.dim()
    for i in range(input.size[dim]): 
        total_slice[dim] = i
        output_tensor += input[tuple(total_slice)]
    return output_tensor


class Size: 
    def __init__(self, sizes: list) -> None: 
        if isinstance(sizes, int): 
            sizes = [sizes]
        _enforce_type(sizes, tuple, int)
        self.data = list(sizes)
    
    def __eq__(self, other: Self) -> bool: 
        _enforce_type(other, Size)
        if self.dim() != other.dim(): 
            return False
        return all([i == j for i, j in zip(self, other)])
    
    def __ne__(self, other: Self) -> bool: 
        _enforce_type(other, Size)
        return not self == other
    
    def __getitem__(self, dim: Union[int, slice]) -> Union[int, list]: 
        return self.data[dim]
    
    def __setitem__(self, dim: int, value: int) -> int: 
        _enforce_type(dim, int)
        _enforce_type(value, int)
        self.data[dim] = value
    
    def __repr__(self): 
        return f"Size({str(self.data)})"
    
    def clone(self): 
        return Size(self.data.copy())
    
    def total(self): 
        prod = 1
        for i in self.data: 
            prod *= i
        return prod
    
    def dim(self): 
        return len(self.data)