import random
import numpy as np
import torch

from eval.GLUE.CoLA.eval import CoLAEval
from eval.math.train.train import MathTrain

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


math_train = MathTrain(low_rank_adaptation="LoRA_HF")
math_train.train()
