import copy
import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.math.common import MathCommon
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import os

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    IGNORE_INDEX = -100

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class MathTrain(MathCommon):

    IGNORE_INDEX = -100

    PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        raw_train_datasets = load_dataset("meta-math/MetaMathQA", split=self.dataset_split)
        train_dataset = raw_train_datasets.map(
            self.train_tokenize_function,
            batched=True,
            batch_size=self.batch_size,
            num_proc=16, # 32
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": self.tokenizer, "query": self.dataset_field[0], "response": self.dataset_field[1]}
        )
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        self.data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        if "CorDA" in self.low_rank_adaptation:
            subset_dataset = torch.utils.data.Subset(train_dataset, range(256))
            subset_dataloader = DataLoader(
                subset_dataset,
                batch_size=1,
                collate_fn=data_collator
            )
            self.model = self.apply_low_rank_adaptation(self.model, corda_method="ipm", calib_loader=subset_dataloader)
        else:
            self.model = self.apply_low_rank_adaptation(self.model)

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def train_tokenize_function(self, examples, tokenizer, query, response):
        sources = [self.PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
        targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
        data_dict = self.preprocess(sources, targets, tokenizer)
        return data_dict

    def train(self):
        training_args = TrainingArguments(
            output_dir=f"results/math/{self.model_name}_{self.low_rank_adaptation}_r{self.lora_r}_checkpoint",
            num_train_epochs=self.num_epochs,
            lr_scheduler_type=self.lr_scheduler,
            warmup_ratio=self.warmup_ratio,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=10,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=self.model, 
            tokenizer=self.tokenizer, 
            args=training_args, 
            **self.data_module
        )
        self.model.config.use_cache = False
        trainer.train()
        trainer.save_state()  # save checkpoint for resuming training

        # saving model for inference
        self.model.save_pretrained(os.path.join('results', 'math', f'{self.model_name}_{self.low_rank_adaptation}_r{self.lora_r}'))
