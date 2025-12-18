# src/utils.py

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)
    print(f"✔ Saved checkpoint to: {path}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
