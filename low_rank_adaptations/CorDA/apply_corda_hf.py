import torch
import torch.nn as nn
from tqdm import tqdm
from peft import get_peft_model, TaskType
from peft.tuners.lora.config import CordaConfig, LoraConfig
from peft.tuners.lora.corda import preprocess_corda

@torch.no_grad()
def run_model(model, calib_loader):
    model.eval()
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

def init_and_apply_corda_hf(
    model, 
    task_type: TaskType,
    calib_loader, 
    corda_method, 
    r: int, 
    alpha: int, 
    dropout: float,
    device: torch.device,
):
    if corda_method != "ipm" and corda_method != "kpm":
        raise ValueError(f"Invalid corda method: {corda_method}")

    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            linear_layer_names.append(name)

    corda_config = CordaConfig(
        corda_method=corda_method,
    )

    lora_config = LoraConfig(
        task_type=task_type,
        init_lora_weights="corda",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=linear_layer_names,
        corda_config=corda_config,
    )

    model.to(device)

    preprocess_corda(model, lora_config, run_model=lambda: run_model(model, calib_loader))


    model = get_peft_model(model, lora_config)

    return model
