import random
import numpy as np
import torch

from eval.GLUE.CoLA.eval import CoLAEval

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


cola_eval = CoLAEval()
cola_eval.train()
cola_eval.eval()
