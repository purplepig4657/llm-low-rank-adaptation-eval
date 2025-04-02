import random
import numpy as np
import torch

from eval import *
import argparse

TASK_TO_CLASS = {
    "cola": CoLAEval,
    "sst2": SST2Eval,
    "mrpc": MRPCEval,
    "sts-b": STSBEval,
    "qnli": QNLIEval,
    "rte": RTEEval,
}

LRA_LIST = [
    "LoRA",
    "PiSSA",
    "CorDA",
    "DoRA",
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add arguments
    parser.add_argument('--eval_tasks', nargs='+', default=TASK_LIST, type=str, 
                       help='Evaluation tasks (space-separated list)')
    parser.add_argument('--low_rank_adaptations', nargs='+', default=LRA_LIST, type=str, 
                       help='Low rank adaptations (space-separated list)')
    parser.add_argument('--lora_r', default=128, type=int, help='LoRA r')
    parser.add_argument('--lora_alpha', default=128, type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.0, type=float, help='LoRA dropout')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=4e-5, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--max_length', default=128, type=int, help='Max length')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    
    return parser.parse_args()

def evaluate_task(task_name: str, low_rank_adaptation: str, args: argparse.Namespace):
    task_class = TASK_TO_CLASS[task_name]
    task = task_class(
        low_rank_adaptation=low_rank_adaptation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    task.train()
    result = task.eval()
    task.cleanup()

    return result

def main():
    args = parse_args()
    set_seed(args.seed)

    for task_name in args.eval_tasks:
        for low_rank_adaptation in args.low_rank_adaptations:
            evaluate_task(task_name, low_rank_adaptation, args)

if __name__ == "__main__":
    main()
