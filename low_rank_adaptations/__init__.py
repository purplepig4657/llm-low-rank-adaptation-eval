from .LoRA.apply_lora import apply_lora
from .PiSSA.apply_pissa import apply_pissa
from .LoRA.apply_lora_hf import apply_lora_hf
from .PiSSA.apply_pissa_hf import apply_pissa_hf
from .DoRA.apply_dora_hf import apply_dora_hf

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || "
        f"trainable%: {100 * trainable_params / all_params:.2f}%"
    )

__all__ = ["apply_lora", "apply_pissa", "apply_lora_hf", "apply_pissa_hf", "apply_dora_hf", "print_trainable_parameters"]
