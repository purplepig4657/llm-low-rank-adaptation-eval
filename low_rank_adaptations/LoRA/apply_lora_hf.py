from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

def apply_lora_hf(model, task_type: TaskType, r=128, alpha=128, dropout=0) -> nn.Module:
    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            linear_layer_names.append(name)

    peft_config = LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=linear_layer_names,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model
