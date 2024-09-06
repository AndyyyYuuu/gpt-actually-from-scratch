from typing import Self 


class Tensor:
    
    def __init__(self, data):
        self.shape = self._detect_shape(data)
        self.data = data

    def _detect_shape(self, data): 
        shape = []
        depth_data = data
        while isinstance(depth_data, (list, tuple)): 
            shape.append(len(depth_data))
            if len(depth_data) == 0: 
                break
            depth_data = depth_data[0]
        return tuple(shape)

    def shape(self) -> tuple: 
        return self.shape
    
    def __repr__(self) -> str: 
        return f"Tensor(shape={self.shape}, data={self.data})"
    
    def __eq__(self, other) -> bool: 
        return self._data_eq(self.data, other.data)
    
    @staticmethod
    def _data_eq(data1, data2): 
        if len(data1) != len(data2): 
            return False
        if isinstance(data1[0], (list, tuple)): 
            return all([Tensor._data_eq(data1[i], data2[i]) for i in range(len(data1))])
        else: 
            return all([data1[i] == data2[i] for i in range(len(data1))])
    
    def __matmul__(self, other) -> Self:
        
        if len(self.data[0]) != len(other.data): 
            if len(other.data[0]) == len(self.data): 
                self, other = other, self
            else: 
                raise ValueError("The number of columns in the first matrix must equal the number of rows in the second")
        n = len(self.data)
        m = len(other.data)
        p = len(other.data[0])
        c = [[0]*p for i in range(n)]
        for i in range(n): 
            for j in range(p): 
                for k in range(m): 
                    c[i][j] += self.data[i][k]*other.data[k][j]
        return Tensor(c)


def _zeros_list(*shape) -> list: 
    if len(shape) == 1: 
        return shape[0] * [0]
    return shape[0] * [_zeros_list(*shape[1:])]

def zeros(*shape): 
    return Tensor(_zeros_list(*shape))