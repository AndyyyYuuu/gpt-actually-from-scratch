from typing import Self, Union, Callable
from .utils import _enforce_type
import builtins
import random
import functools

class Tensor:

    def __init__(self, data: list, size: 'Size', stride: tuple): 
        
        _enforce_type(data, list, float)
        _enforce_type(size, Size)
        _enforce_type(stride, tuple, float)
        self.shape = size
        self.data = data
        self.stride = stride
    

    def __getitem__(self, index: Union[int, slice, list[Union[int, slice]]]) -> Union[float, list]: 
        if isinstance(index, (int, slice)): 
            index = (index,)
        
        _enforce_type(index, tuple)
        index = list(index)
        
        if all([isinstance(i, int) for i in index]) and len(index) == self.shape.dim():
            flat_index = builtins.sum([i*j for i, j in zip(self.stride, list(index))])
            return self.data[flat_index]
        
        if len(index) > self.shape.dim(): 
            raise IndexError(f"Tensor index out of range (expected {self.shape.dim()} dimensions, found {len(index)})")
        while len(index) < self.shape.dim(): 
            index.append(slice(None))
        

        one_dims = [] # keep track of which dimensions will be squeezed
        for i in range(len(index)): 
            if isinstance(index[i], int): 
                index[i] = slice(index[i], index[i]+1, None)
                one_dims.append(i)
            elif not isinstance(index[i], slice): 
                raise TypeError(f"Tensor index must be list of int or slice (found {type(index[i])})")
            
        result = zeros(*[len(range(self.shape[i])[index[i]]) for i in range(len(index))])
        

        def _get_slice(slices: list[slice], pos_self: list[int], pos_result: list[int], depth=0) -> None: 
            if len(slices) == 0: 
                result[pos_result] = self[pos_self]
            else: 
                for i,j in enumerate(range(self.shape[depth])[slices[0]]): 
                    _get_slice(slices[1:], pos_self+[j], pos_result+[i], depth+1)
        
        _get_slice(index, [], [])

        # Squeeze sliced dimensions
        for i, j in enumerate(one_dims): 
            result.squeeze(j-i) # -i to calibrate index after removing dimensions

        return result


    def __setitem__(self, index: list, value) -> None: 
        _enforce_type(index, tuple)
        flat_index = builtins.sum([i*j for i, j in zip(self.stride, index)])
        self.data[flat_index] = value

    def clone(self) -> Self: 
        return Tensor(self.data.copy(), self.shape.clone(), self.stride.copy())

    def size(self) -> 'Size': 
        return self.shape
    
    def __repr__(self) -> str: 
        return f"Tensor(shape={self.shape}, data={self.tolist()})"
    
    def __eq__(self, other: Self) -> bool: 
        _enforce_type(other, Tensor)
        return other.shape == self.shape and all([i==j for i, j in zip(self.data, other.data)])

    @staticmethod
    def elementwise(op: Callable): 
        @functools.wraps(op)
        def _element_wise(t_self: Self, t_other: Self) -> Callable:
            if isinstance(t_other, Tensor) and t_other.shape == Size([]):
                t_other = t_other[0]
            if isinstance(t_other, (float, int)): 
                result = t_self.clone()
                for i in range(len(result.data)): 
                    result.data[i] += t_other
                
                return result

            if t_self.shape != t_other.shape: 

                t_self = t_self.clone()
                _enforce_type(t_other, Tensor)
                t_other = t_other.clone()
                while t_other.shape.dim() < t_self.shape.dim(): 
                    t_other.unsqueeze(0)
                while t_other.shape.dim() > t_self.shape.dim(): 
                    t_self.unsqueeze(0)
                
                for i, j in zip(t_self.shape, t_other.shape): 
                    if i != j and i != 1 and j != 1:
                        raise ValueError(f"Incompatible element-wise operation for tensors of shapes {t_self.shape} and {t_other.shape}.")
                
                if t_self.shape.total() < t_other.shape.total(): 
                    t_self.expand(t_other.shape)
                else: 
                    t_other.expand(t_self.shape)

            result = zeros(*t_self.shape)
            def _op_tensors(t1, t2, output, index=[]): 
                if len(index) == t1.shape.dim(): 
                    output[index] = op(t1[index], t2[index])
                else: 
                    for i in range(t1.shape[len(index)]): 
                        _op_tensors(t1, t2, output, index + [i])
            _op_tensors(t_self, t_other, result)
            return result
        return _element_wise

    @elementwise
    def __add__(t1: float, t2: float) -> float: 
        return t1 + t2
    
    @elementwise
    def __radd__(t1: float, t2: float) -> float: 
        return t2 + t1
    
    @elementwise
    def __mul__(t1: float, t2: float) -> float: 
        return t1 * t2

    @elementwise
    def __rmul__(t1: float, t2: float) -> float: 
        return t2 * t1
    
    @elementwise
    def __truediv__(t1: float, t2: float) -> float: 
        return t1 / t2
    
    @elementwise
    def __rtruediv__(t1: float, t2: float) -> float: 
        return t2 / t1
    
    @elementwise
    def __floordiv__(t1: float, t2: float) -> float: 
        return t1 / t2
    
    @elementwise
    def __rfloordiv__(t1: float, t2: float) -> float: 
        return t2 / t1
    
    @elementwise
    def __pow__(t1: float, t2: float) -> float: 
        return t1 ** t2
    
    @elementwise
    def __rpow__(t1: float, t2: float) -> float: 
        return t2 ** t1
    
    @elementwise
    def __neg__(t1: float, t2: float) -> float: 
        return t2 ** t1
    
    @elementwise
    def __lt__(t1: float, t2: float) -> bool: 
        return t1 < t2
    
    @elementwise
    def __gt__(t1: float, t2: float) -> bool: 
        return t1 > t2
    
    @elementwise
    def __le__(t1: float, t2: float) -> bool: 
        return t1 >= t2
    
    @elementwise
    def __ge__(t1: float, t2: float) -> bool: 
        return t1 <= t2
    
    @elementwise
    def __eq__(t1: float, t2: float) -> bool: 
        return t1 == t2
    
    @elementwise
    def __ne__(t1: float, t2: float) -> bool: 
        return t1 != t2
    
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
        _other = other.clone()

        # Whether inputs came with batch dimension: used to determine whether to output with batch dimension
        has_batch = _self.shape.dim() >= 3 or _other.shape.dim() >= 3 
        
        
        while _self.shape.dim() < 3: 
            _self.unsqueeze(0)
        if _other.shape.dim() < 3: 
            _other.unsqueeze(0)
        
        if _self.shape[2] != _other.shape[1]: 
            if _other.shape[2] == _self.shape[1]: 
                _self, _other = _other, _self
            else: 
                raise ValueError(f"The number of columns in the first matrix must equal the number of rows in the second (found {_self.shape} and {other.shape})")
        if _self.shape[0] != _other.shape[0] and _self.shape[0] != 1 and _other.shape[0] != 1: 
            raise ValueError(f"Incompatible batch sizes in batch matmul: {_self.shape[0]} and {_other.shape[0]}")
        
        self_batch_1 = _self.shape[0] == 1    # Whether batch size is 1 for each tensor
        other_batch_1 = _other.shape[0] == 1  # Used to determine whether or not to iterate through batch dimension during loop
        
        batch_size = _self.shape[0] if _self.shape[0] > 1 else _other.shape[0]

        n = _self.shape[1]
        m = _other.shape[1]
        p = _other.shape[2]
        c = zeros(batch_size, n, p)

        for b in range(batch_size): 
            for i in range(n): 
                for j in range(p): 
                    for k in range(m): 
                        c[b, i, j] += _self[(0 if self_batch_1 else b), i, k]*_other[(0 if other_batch_1 else b), k, j]
        
        if c.shape[0] == 1 and not has_batch: 
            c.squeeze(0)
        return c
    
    def __round__(self) -> Self: 
        result = self.clone()
        for i in range(len(result.data)): 
            result.data[i] = round(result.data[i])
        return result
    
    def relu(self) -> Self: 
        result = self.clone()
        for i in range(len(result.data)): 
            if result.data[i] < 0: 
                result.data[i] = 0 
        return result
    
    def tolist(self) -> Union[list, float]: 
        def build_list(dim, index): 
            if dim == self.shape.dim(): 
                return self[index]
            return [build_list(dim + 1, index + [i]) for i in range(self.shape[dim])]
        return build_list(0, [])

    def unsqueeze(self, dim: int) -> None: 
        _enforce_type(dim, int)
        self.shape.data.insert(dim, 1)
        if dim != 0: 
            self.stride.insert(dim, self.stride[dim-1])
        else: 
            self.stride.insert(dim, 1)
    
    def squeeze(self, dim: int=None) -> None: 
        if dim is None: 
            new_size = []
            new_stride = []
            for i in range(self.shape.dim()): 
                if self.shape[i] != 1: 
                    new_size.append(self.shape[i])
                    new_stride.append(self.stride[i])
            self.shape = Size(new_size)
            self.stride = new_stride
        else: 
            _enforce_type(dim, int)
            if self.shape[dim] == 1:
                self.stride.pop(dim)
                self.shape.data.pop(dim)
            else: 
                raise ValueError(f"squeeze() expected a dimension of size 1 at index {dim}, got {self.shape[dim]}")
            
    def expand(self, size: 'Size') -> None:
        _enforce_type(size, Size)
        for i in range(self.shape.dim()): 
            if self.shape[i] > size[i]:
                raise ValueError(f"Tensor of size {self.shape} is too large to be expanded to {size}.")
            elif self.shape[i] < size[i]: 
                if self.shape[i] == 1:
                    self.shape[i] = size[i]
                    self.stride[i] = 0
                else: 
                    raise ValueError(f"Tensor of size {self.shape} cannot be expanded to {size}.")

    
def transpose(input: Tensor, dim1: int, dim2: int) -> Tensor: 
    _enforce_type(input, Tensor)
    _enforce_type(dim1, int)
    _enforce_type(dim2, int)
    output = input.clone()
    output.shape[dim1], output.shape[dim2] = input.shape[dim2], input.shape[dim1]
    output.stride[dim1], output.stride[dim2] = input.stride[dim2], input.stride[dim1]
    return output
    

def cat(tensors: tuple[Tensor, ...], dim: int=0): 
    _enforce_type(tensors, tuple, Tensor)
    _enforce_type(dim, int)
    for t in tensors[1:]: 
        for i in range(t.shape.dim()): 
            if i != dim and t.shape[i] != tensors[0].shape[i]: 
                raise ValueError(f"Tensors of sizes {tensors[0].shape, t.shape} cannot be concatenated.")
    tensor_dims = tensors[0].shape.dim()
    new_shape = [tensors[0].shape[i] if i != dim else builtins.sum([t.shape[i] for t in tensors]) for i in range(tensor_dims)]
    result = zeros(*new_shape)

    offset = 0  # Offset value for copying: sum of previous tensor lengths
    for i in range(len(tensors)): 
        def _copy_sublist(pos: list, depth: int): 
            if depth == 0: 
                result_pos = pos.copy()
                result_pos[dim] += offset
                result[result_pos] = tensors[i][pos]
            else: 
                for j in range(tensors[i].shape[-depth]): 
                    _copy_sublist(pos+[j], depth-1)
        _copy_sublist([], tensor_dims)
        offset += tensors[i].shape[dim]
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
    elif isinstance(data, (int, float)): 
        data = [data]
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
    for i in tensor.shape: 
        prod*=i
    flat_tensor.shape = Size([prod])
    flat_tensor.stride = [1]
    return flat_tensor


def _num_tensor(size: 'Size', num: Union[int, float]) -> Tensor: 
    return Tensor(size.total()*[num], size, _get_stride(size))

def zeros(*shape: Union[tuple, list]) -> Tensor: 
    _enforce_type(shape, tuple, int)
    return _num_tensor(Size(shape), num=0)

def ones(*shape: Union[tuple, list]) -> Tensor:
    _enforce_type(shape, tuple, int)
    return _num_tensor(Size(shape), num=1)

def rand(*shape: Union[tuple, list]) -> Tensor:
    _enforce_type(shape, tuple, int)
    shape = Size(shape)
    return Tensor([random.getrandbits(1) for i in range(shape.total())], shape, _get_stride(shape))

def randn(*shape: Union[tuple, list]) -> Tensor:
    _enforce_type(shape, tuple, int)
    shape = Size(shape)
    return Tensor([random.normalvariate() for i in range(shape.total())], shape, _get_stride(shape))


def sum(input: Tensor, dim:int=None, keepdim:bool=False) -> Union[float, Tensor]: 
    _enforce_type(input, Tensor)
    _enforce_type(keepdim, bool)
    
    if dim is None: 
        return tensor(builtins.sum(input.data))

    _enforce_type(dim, int)

    if dim >= input.shape.dim(): 
        raise ValueError(f"dimension {dim} out of range for {input.shape.dim()}-d Tensor")
    
    if input.shape.dim() == 1: 
        return tensor(builtins.sum(input.data))
    
    if keepdim: 
        output_tensor = zeros(*(input.shape[:dim]+[1,]+input.shape[dim+1:]))
    else:
        output_tensor = zeros(*(input.shape[:dim]+input.shape[dim+1:]))

    total_slice = [slice(None)] * input.shape.dim()
    for i in range(input.shape[dim]): 
        if keepdim: 
            total_slice[dim] = slice(i, i+1, None)
        else: 
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