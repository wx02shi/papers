import torch
from torch import nn
from bitnet158 import BitLinear158

def inject(model, copy_weights = True, module_class=BitLinear158):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            new_module = module_class(module.in_features, module.out_features, module.bias is not None)
            if copy_weights:
                new_module.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, nn.Module) and name != "":
            inject(module, copy_weights, module_class)
    return model