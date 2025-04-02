import torch
import torch.nn as nn
from tqdm import tqdm
from peft import get_peft_model
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
    calib_loader, 
    corda_method: "ipm" | "kpm", 
    r: int = 128, 
    alpha: int = 128, 
    dropout: float = 0.0,
):
    if corda_method != "ipm" and corda_method != "kpm":
        raise ValueError(f"Invalid corda method: {corda_method}")

    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layer_names.append(name)

    corda_config = CordaConfig(
        corda_method=corda_method,
    )

    lora_config = LoraConfig(
        init_lora_weights="corda",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=linear_layer_names,
        corda_config=corda_config,
    )

    preprocess_corda(model, lora_config, run_model=lambda: run_model(model, calib_loader))


    model = get_peft_model(model, lora_config)

    return model
