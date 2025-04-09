import multiprocessing
import random
import numpy as np
import torch

from eval.math.train import MathTrain
from eval.math.MATH.eval import MATHEval

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


# math_train = MathTrain(low_rank_adaptation="LoRA_HF", lora_r=128, lora_alpha=128)
# math_train.train()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    math_eval = MATHEval(model="./results/math/meta-llama/Meta-Llama-3-8B_LoRA_HF_r128", data_path="/root/thk_tmp/llm-low-rank-adaptation-eval/eval/math/MATH/data/MATH_test_small.jsonl", batch_size=50)
    acc = math_eval.eval()
    print(acc)
