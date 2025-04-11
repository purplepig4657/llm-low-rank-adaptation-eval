import random
import numpy as np
import torch
from eval.math.train import MathTrain
import argparse

LRA_LIST = [
    # "full",
    "LoRA_HF",
    "PiSSA_HF",
    "CorDA_HF",
    "DoRA_HF",
]


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add arguments
    parser.add_argument('--low_rank_adaptations', nargs='+', default=LRA_LIST, type=str, 
                       help='Low rank adaptations (space-separated list)')
    parser.add_argument('--lora_r', default=128, type=int, help='LoRA r')
    parser.add_argument('--lora_alpha', default=128, type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.0, type=float, help='LoRA dropout')
    parser.add_argument('--num_epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--model_name', default="meta-llama/Meta-Llama-3-8B", type=str, help='Model name')
    parser.add_argument('--model_max_length', default=512, type=int, help='Model max length')
    parser.add_argument('--dataset_split', default="train[:100000]", type=str, help='Dataset split')
    parser.add_argument('--optim', default="adamw_torch", type=str, help='Optimizer')
    parser.add_argument('--dataset_field', default=["query", "response"], type=list, help='Dataset field')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('--warmup_ratio', default=0.03, type=float, help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', default=128, type=int, help='Gradient accumulation steps')
    parser.add_argument('--lr_scheduler', default="cosine", type=str, help='LR scheduler')
    
    return parser.parse_args()

if __name__ == "__main__":
    set_seed(0)
    args = parse_args()

    for lra in args.low_rank_adaptations:
        math_train = MathTrain(
            low_rank_adaptation=lra, 
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            model_name=args.model_name,
            model_max_length=args.model_max_length,
            dataset_split=args.dataset_split,
            optim=args.optim,
            dataset_field=args.dataset_field,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr_scheduler=args.lr_scheduler,
        )
        math_train.train()
        math_train.cleanup()
