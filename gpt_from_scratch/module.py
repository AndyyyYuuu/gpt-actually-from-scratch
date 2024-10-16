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
    
    def __setattr__(self, name: str, value: any) -> None:
        if isinstance(value, Module):
            self._modules[name] = value
            sub_params = value.parameters()
            new_dict = {}
            for param_name, param_value in value.parameters().items():
                new_dict[f"{name}.{param_name}"] = param_value
            self.parameters = new_dict | self.parameters
            
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        
        super(Module, self).__setattr__(name, value)
    
    def forward(self): 
        raise NotImplementedError("method `forward` not implemented for subclass of Module")
    
    def register_parameter(self, name: str, value: Parameter) -> None:
        self._parameters[name] = value
    
    def parameters(self) -> dict:
        return self._parameters
    
    def train(self): 
        for _, submodule in self._modules.items(): 
            submodule.train()
        self.training = True

    def eval(self): 
        for _, submodule in self._modules.items(): 
            submodule.eval()
        self.training = False


class Linear(Module): 

    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(tensor.ones(in_features, out_features))
        self.bias = Parameter(tensor.ones(out_features))
        #self.register_parameter("weight", self.weight)
        #self.register_parameter("bias", self.bias)

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

