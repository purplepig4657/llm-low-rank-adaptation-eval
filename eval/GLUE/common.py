import torch
from low_rank_adaptations import apply_lora, print_trainable_parameters
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

    def apply_low_rank_adaptation(self, model):
        if self.low_rank_adaptation == "LoRA":
            model = apply_lora(
                model, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            )
        elif self.low_rank_adaptation == "LoRA_HF":
            self.linear_layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    self.linear_layer_names.append(name)

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, peft_config)
        else:
            raise ValueError(f"Invalid or not supported low rank adaptation: {self.low_rank_adaptation}")

        print_trainable_parameters(model)

        return model
