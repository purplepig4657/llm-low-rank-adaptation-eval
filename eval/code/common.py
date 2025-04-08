import torch
from low_rank_adaptations import apply_lora, apply_pissa, apply_lora_hf, apply_pissa_hf, apply_dora_hf, init_and_apply_corda_hf, print_trainable_parameters
from peft import TaskType
import gc

class CodeCommon:
    def __init__(
        self,
        model_name = "meta-llama/Meta-Llama-3-8B", 
        model_max_length = 512,
        dataset_split = "train[:100000]",
        optim = "adamw_torch",
        dataset_field = ["query", "answer"],
        num_epochs = 1,
        weight_decay = 0.0,
        warmup_ratio = 0.03,
        learning_rate = 2e-5,
        batch_size = 1,
        gradient_accumulation_steps = 128,
        seed = 0,
        low_rank_adaptation = "LoRA",
        lora_r = 128,
        lora_alpha = 128,
        lora_dropout = 0.0,
        lr_scheduler = "cosine",
        device = "cuda" if torch.cuda.is_available() else "cpu",
        apply_lra = True,
    ):
        self.model_name = model_name
        self.model_max_length = model_max_length
        self.dataset_split = dataset_split
        self.optim = optim
        self.dataset_field = dataset_field
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.low_rank_adaptation = low_rank_adaptation
        self.lora_r = lora_r
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.apply_lra = apply_lra

    def apply_low_rank_adaptation(self, model, corda_method = "ipm", calib_loader=None):
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
                model, 
                TaskType.CAUSAL_LM,
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
                model, 
                TaskType.CAUSAL_LM,
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "DoRA_HF":
            model = apply_dora_hf(
                model, 
                TaskType.CAUSAL_LM,
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "CorDA_HF":
            if calib_loader is None:
                raise ValueError("Calibration loader is required for CorDA_HF")

            model = init_and_apply_corda_hf(
                model, 
                TaskType.CAUSAL_LM,
                calib_loader, 
                corda_method=corda_method, 
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                device=self.device,
            )
        elif self.low_rank_adaptation == "full":
            pass
        else:
            raise ValueError(f"Invalid or not supported low rank adaptation: {self.low_rank_adaptation}")

        print_trainable_parameters(model)

        return model

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
            
        gc.collect()

