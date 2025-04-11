import os
import subprocess
from itertools import product
import argparse

LRA_TO_MODEL_PATH = {
    "LoRA_HF": "results/code/meta-llama/Meta-Llama-3-8B_LoRA_HF_r32",
    "PiSSA_HF": "results/code/meta-llama/Meta-Llama-3-8B_PiSSA_HF_r32",
    "DoRA_HF": "results/code/meta-llama/Meta-Llama-3-8B_DoRA_HF_r32",
    "CorDA_HF": "results/code/meta-llama/Meta-Llama-3-8B_CorDA_HF_r32",
}

tasks = [
    "humaneval",
    "mbpp",
]

base_command = [
    "accelerate", "launch", "bigcode-evaluation-harness/main.py",
    "--batch_size", "50",
    "--allow_code_execution",
    "--save_generations"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Code evaluation script')
    # Add arguments
    parser.add_argument('--eval_tasks', nargs='+', default=tasks, type=str, 
                       help='Evaluation tasks (space-separated list)')
    parser.add_argument('--low_rank_adaptations', nargs='+', default=LRA_TO_MODEL_PATH.keys(), type=str, 
                       help='Low rank adaptations (space-separated list)')
    parser.add_argument('--bigcode_eval_path', default="bigcode-evaluation-harness", type=str,
                        help='Path to the bigcode evaluation harness')
    
    # Add arguments for each model path
    parser.add_argument('--lora_model_path', default=LRA_TO_MODEL_PATH["LoRA_HF"], type=str,
                        help='Path to the LoRA model')
    parser.add_argument('--pissa_model_path', default=LRA_TO_MODEL_PATH["PiSSA_HF"], type=str,
                        help='Path to the PiSSA model')
    parser.add_argument('--corda_model_path', default=LRA_TO_MODEL_PATH["CorDA_HF"], type=str,
                        help='Path to the CorDA model')
    parser.add_argument('--dora_model_path', default=LRA_TO_MODEL_PATH["DoRA_HF"], type=str,
                        help='Path to the DoRA model')
    return parser.parse_args()

def evaluate_task(task_name: str, lra: str):
    print(f"Running: model={LRA_TO_MODEL_PATH[lra]} task={task_name}")
    command = base_command + [
        "--model", LRA_TO_MODEL_PATH[lra],
        "--tasks", task_name,
        "--metric_output_path", f"{task_name}_{lra}.json"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed for model={LRA_TO_MODEL_PATH[lra]} task={task_name}: {e}")

if __name__ == "__main__":
    args = parse_args()

    LRA_TO_MODEL_PATH = {
        "LoRA_HF": args.lora_model_path,
        "PiSSA_HF": args.pissa_model_path,
        "CorDA_HF": args.corda_model_path,
        "DoRA_HF": args.dora_model_path,
    }

    base_command[2] = os.path.join(args.bigcode_eval_path, "main.py")

    for task_name in args.eval_tasks:
        for low_rank_adaptation in args.low_rank_adaptations:
            evaluate_task(task_name, low_rank_adaptation)
