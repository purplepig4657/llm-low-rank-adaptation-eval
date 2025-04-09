import random
import numpy as np
import torch

from eval.math.train import MathTrain
from eval.math.MATH.eval import MathEval

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


# math_train = MathTrain(low_rank_adaptation="DoRA_HF", lora_r=32, lora_alpha=32)
# math_train.train()

math_eval = MathEval(model="./results/math/meta-llama/Meta-Llama-3-8B_CorDA_HF_r32", data_path="eval/math/MATH/math_data.jsonl")
acc = math_eval.eval()
print(acc)
