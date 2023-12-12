import os
import torch
import numpy as np
import random

SLURM_OUTPUT_DIR = f"J{os.getenv('SLURM_JOB_ID')}_{os.getenv('SLURM_JOB_NAME')}"

def seed_all(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
