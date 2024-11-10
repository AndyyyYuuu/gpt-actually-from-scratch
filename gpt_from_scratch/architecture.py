from . import functional as F
from . import tensor, module
from .parameter import Parameter
from .utils import _enforce_type


class AttentionHead(module.Module): 

    def __init__(self, head_size: int, n_embed: int) -> None:
        super().__init__()
        self.key = module.Linear(n_embed, head_size, bias=False)
        self.query = module.Linear(n_embed, head_size, bias=False)
        self.value = module.Linear(n_embed, head_size, bias=False)
        self.dropout = module.Dropout(0.2)

    def forward(self, x): 
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out