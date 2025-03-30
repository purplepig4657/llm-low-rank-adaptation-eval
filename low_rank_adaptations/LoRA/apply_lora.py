import torch.nn as nn
from .loralib import Linear as LoRALinear, LoRALayer

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError

def apply_lora(model, r=128, alpha=128, dropout=0) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            sub_names = name.split('.')
            for sub_name in sub_names[:-1]:
                parent = getattr(parent, sub_name)
            old_linear = getattr(parent, sub_names[-1])

            # LoRA.Linear로 교체
            lora_linear = LoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=old_linear.bias is not None
            )
            lora_linear.weight.data = old_linear.weight.data.clone()
            if old_linear.bias is not None:
                lora_linear.bias.data = old_linear.bias.data.clone()
            setattr(parent, sub_names[-1], lora_linear)
    mark_only_lora_as_trainable(model)
    return model
