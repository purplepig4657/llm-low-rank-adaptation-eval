import torch
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm

import numpy as np
from datasets import load_dataset
import evaluate

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from low_rank_adaptations import apply_lora, print_trainable_parameters

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CoLAEval:
    DATASET_NAME = "glue"
    TASK_NAME = "cola"

    def __init__(
            self, 
            model_name: str = "roberta-base", 
            low_rank_adaptation: str = "LoRA",
            lora_r: int = 128,
            lora_alpha: int = 128,
            lora_dropout: float = 0.0,
            lora_target_modules: list[str] = None,
            num_epochs: int = 3,
            learning_rate: float = 4e-5,
            batch_size: int = 32,
            max_length: int = 128,
            lr_scheduler: str = "linear",
        ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.num_epochs = num_epochs
        self.max_length = max_length
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size

        if low_rank_adaptation == "LoRA":
            self.model = apply_lora(self.model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        elif low_rank_adaptation == "LoRA_HF":
            self.linear_layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    self.linear_layer_names.append(name)

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
        else:
            raise ValueError(f"Invalid or not supported low rank adaptation: {low_rank_adaptation}")

        print_trainable_parameters(self.model)

        self.dataset = load_dataset(self.DATASET_NAME, self.TASK_NAME)

        self.cola_metric = evaluate.load(self.DATASET_NAME, self.TASK_NAME)

        self.tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True, remove_columns=["sentence", "idx"])

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, return_tensors="pt")

        self.train_dataloader = DataLoader(self.tokenized_dataset["train"], shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)
        self.eval_dataloader = DataLoader(self.tokenized_dataset["validation"], batch_size=self.batch_size, collate_fn=self.data_collator)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)


        self.num_training_steps = self.num_epochs * len(self.train_dataloader)

        self.lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def tokenize_function(self, example):
        return self.tokenizer(
            example["sentence"], 
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.cola_metric.compute(predictions=predictions, references=labels)

    def train(self):
        self.model.to(self.device)

        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train(True)
        for _ in range(self.num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

    def eval(self):
        self.model.train(False)
        self.model.eval()
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            self.cola_metric.add_batch(predictions=predictions, references=batch["labels"])

        result = self.cola_metric.compute()
        print(result)
