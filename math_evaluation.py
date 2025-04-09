import random
import numpy as np
import torch

from eval import *
import argparse

from eval.math.MATH.eval import MATHEval
from eval.math.GSM8K.eval import GSM8KEval

TASK_TO_CLASS = {
    "math": MATHEval,
    "gsm8k": GSM8KEval,
}

LRA_TO_MODEL_PATH = {
    # "full": "results/math/meta-llama/Meta-Llama-3-8B",
    "LoRA_HF": "results/math/meta-llama/Meta-Llama-3-8B_LoRA_HF_r32",
    "PiSSA_HF": "results/math/meta-llama/Meta-Llama-3-8B_PiSSA_HF_r32",
    "CorDA_HF": "results/math/meta-llama/Meta-Llama-3-8B_CorDA_HF_r32",
    "DoRA_HF": "results/math/meta-llama/Meta-Llama-3-8B_DoRA_HF_r32",
}

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
    parser.add_argument('--eval_tasks', nargs='+', default=TASK_TO_CLASS.keys(), type=str, 
                       help='Evaluation tasks (space-separated list)')
    parser.add_argument('--low_rank_adaptations', nargs='+', default=LRA_TO_MODEL_PATH.keys(), type=str, 
                       help='Low rank adaptations (space-separated list)')
    return parser.parse_args()

def evaluate_task(task_name: str, model_path: str):
    task_class = TASK_TO_CLASS[task_name]
    task = task_class(model_path)
    result = task.eval()
    task.cleanup()
    return result

def main():
    args = parse_args()
    results = []
    for task_name in args.eval_tasks:
        for low_rank_adaptation in args.low_rank_adaptations:
            result = evaluate_task(task_name, LRA_TO_MODEL_PATH[low_rank_adaptation])
            results.append({
                "task": task_name,
                "adaptation": low_rank_adaptation,
                "result": result
            })

    with open("math_evaluation_results.txt", "w") as f:
        f.write(f"Math Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for res in results:
            f.write(f"Task: {res['task']}\n")
            f.write(f"Low-Rank Adaptation: {res['adaptation']}\n")
            f.write(f"Result: {res['result']}\n")

if __name__ == "__main__":
    main()
