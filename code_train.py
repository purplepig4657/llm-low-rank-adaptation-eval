import random
import numpy as np
import torch
from eval.code.train import CodeTrain

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

set_seed(0)


for lra in LRA_LIST:
    code_train = CodeTrain(low_rank_adaptation=lra, lora_r=32, lora_alpha=32)
    code_train.train()
    code_train.cleanup()
