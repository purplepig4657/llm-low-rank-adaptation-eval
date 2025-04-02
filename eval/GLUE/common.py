import torch
from torch import nn
from low_rank_adaptations import apply_lora, apply_pissa, apply_lora_hf, apply_pissa_hf, apply_dora_hf, init_and_apply_corda_hf, print_trainable_parameters
from peft import get_peft_model, LoraConfig, TaskType

class GLUEEvalCommon:
    def __init__(
            self, 
            model_name: str = "roberta-base", 
            low_rank_adaptation: str = "LoRA", 
            lora_r: int = 128, 
            lora_alpha: int = 128, 
            lora_dropout: float = 0.0, 
            num_epochs: int = 3, 
            learning_rate: float = 4e-5, 
            batch_size: int = 32, 
            max_length: int = 128, 
            lr_scheduler: str = "linear",
            apply_lra: bool = True,
            seed: int = 42,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ):
        self.model_name = model_name
        self.low_rank_adaptation = low_rank_adaptation
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr_scheduler = lr_scheduler
        self.seed = seed
        self.device = device
        self.apply_lra = apply_lra

    def apply_low_rank_adaptation(self, model, corda_method: "ipm" | "kpm" = "ipm", calib_loader=None):
        if not self.apply_lra:
            return model

        if self.low_rank_adaptation == "LoRA":
            model = apply_lora(
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "LoRA_HF":
            model = apply_lora_hf(
                TaskType.SEQ_CLS,
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "PiSSA":
            model = apply_pissa(
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "PiSSA_HF":
            model = apply_pissa_hf(
                TaskType.SEQ_CLS,
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "DoRA_HF":
            model = apply_dora_hf(
                TaskType.SEQ_CLS,
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "CorDA_HF":
            if calib_loader is None:
                raise ValueError("Calibration loader is required for CorDA_HF")

            model = init_and_apply_corda_hf(
                TaskType.SEQ_CLS,
                model, 
                calib_loader, 
                corda_method=corda_method, 
            )
        else:
            raise ValueError(f"Invalid or not supported low rank adaptation: {self.low_rank_adaptation}")

        print_trainable_parameters(model)

        return model
