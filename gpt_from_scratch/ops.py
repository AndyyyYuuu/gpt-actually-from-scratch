


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
    
    def __matmul__(self, other):
        n = len(self.data)
        m = len(other.data)
        p = len(other.data[0])
        c = [[0]*p for i in range(n)]
        for i in range(n): 
            for j in range(p): 
                for k in range(m): 
                    c[i][j] += self.data[i][k]*other.data[k][j]
        return c
                
