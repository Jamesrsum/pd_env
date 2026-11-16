"""LoRA's W x + (alpha/r) B A x, with frozen pretrained weights.

Equation/initialization: Hu et al., arXiv:2106.09685. The PD rank/alpha and
target choices are project hypotheses, not a claimed PD paper reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16., dropout=0.):
        super().__init__()
        if not isinstance(base,nn.Linear) or not 0<rank<=min(base.in_features,base.out_features):
            raise ValueError('LoRA requires a linear layer and valid positive rank')
        if alpha<=0 or not 0<=dropout<1:raise ValueError('Invalid LoRA scaling/dropout')
        self.base=base;self.base.requires_grad_(False)
        self.rank=rank;self.scaling=alpha/rank;self.dropout=nn.Dropout(dropout)
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,x):
        return self.base(x)+F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)*self.scaling


def inject_lora(backbone, suffixes, rank=8, alpha=16., dropout=0.):
    matched=[(name,module) for name,module in backbone.named_modules()
             if any(name.endswith(s) for s in suffixes) and isinstance(module,nn.Linear)]
    if not matched:raise ValueError('No requested LoRA targets exist')
    for suffix in suffixes:
        if not any(name.endswith(suffix) for name,module in matched):raise ValueError(f'Missing LoRA target: {suffix}')
    backbone.requires_grad_(False)
    for name,module in matched:
        parent,_,leaf=name.rpartition('.')
        setattr(backbone.get_submodule(parent) if parent else backbone,leaf,LoRALinear(module,rank,alpha,dropout))
    return [name for name,module in matched]


class StandardizedLinear(nn.Module):
    """Exactly preserve a train-fitted sklearn scaler and binary logistic probe."""
    def __init__(self,mean,scale,coef,intercept):
        super().__init__()
        mean=torch.as_tensor(mean,dtype=torch.float32);scale=torch.as_tensor(scale,dtype=torch.float32)
        if mean.ndim!=1 or scale.shape!=mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not scale.gt(0).all():raise ValueError('Invalid frozen train-fitted scaler')
        self.register_buffer('mean',mean.clone());self.register_buffer('scale',scale.clone())
        self.linear=nn.Linear(len(mean),1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(coef,dtype=torch.float32).reshape(1,-1))
            self.linear.bias.copy_(torch.as_tensor(intercept,dtype=torch.float32).reshape(1))

    def forward(self,x):return self.linear((x-self.mean)/self.scale)
