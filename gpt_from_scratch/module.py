from . import functional as F
from . import tensor
from .parameter import Parameter
from .utils import _enforce_type

class Module: 

    def __init__(self): 
        self._parameters = {}
        self._modules = {}
        self.training = False

    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs)
    
    def __setattr__(self, name: str, value: any, base: bool=True) -> None:

        if isinstance(value, Module):
            self._modules[name] = value
            new_dict = {}
            for param_name, param_value in value.state_dict().items():
                new_dict[f"{name}.{param_name}"] = param_value
            self._parameters = new_dict | self._parameters

        elif isinstance(value, (list, tuple)): 
            for i, v in enumerate(value): 
                self.__setattr__(f"{name}.{i}", v, base=False)
        
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        
        if base: 
            super(Module, self).__setattr__(name, value)
    
    def forward(self): 
        raise NotImplementedError("method `forward` not implemented for subclass of Module")
    
    def register_parameter(self, name: str, value: Parameter) -> None:
        self._parameters[name] = value
    
    def state_dict(self) -> dict:
        return self._parameters
    
    def load_state_dict(self, state_dict: dict) -> None:
        def load_recur(_data: dict, prefix: str="") -> None: 
            for name, value in _data.items():
                full_name = prefix + name
                if full_name in state_dict: 
                    if _data[full_name].shape == value.shape:
                        self._parameters[full_name].copy(value)
                    else:
                        raise ValueError(f"Shape mismatch for '{full_name}' in state_dict: "
                                         f"expected {value.shape}, got {value[full_name].shape}")
                else: 
                    print(f"Warning: \"{full_name}\" not found in state_dict")

        _enforce_type(state_dict, dict)
        load_recur(state_dict)
        '''
        for key, value in state_dict.items(): 
            split_key = key.split(".")
            if len(split_key) == 1 and isinstance(value, Parameter): 
                super(Module, self).__setattr__(key, value)
        '''
        


        
    def train(self): 
        for _, submodule in self._modules.items(): 
            submodule.train()
        self.training = True

    def eval(self): 
        for _, submodule in self._modules.items(): 
            submodule.eval()
        self.training = False


class Sequential(Module): 
    def __init__(self, modules: list): 
        super().__init__()
        _enforce_type(modules, tuple, Module)
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x



class Linear(Module): 

    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tensor.randn(in_features, out_features))
        self.bias = Parameter(tensor.randn(out_features))

    def forward(self, x): 
        return F.linear(x, self.weight, self.bias)
    

class Dropout(Module): 

    def __init__(self, p:float=0.5): 
        super().__init__()
        self.p = p
    
    def forward(self, x: tensor.Tensor): 
        _enforce_type(x, tensor.Tensor)

        x = x.clone()

        if self.training: 
            x += tensor.rand(*x.shape())
            x /= 1-self.p


        return x

class ReLU(Module): 

    def __init__(self): 
        super().__init__()
    
    def forward(self, x: tensor.Tensor): 
        _enforce_type(x, tensor.Tensor)
        return F.relu(x)
