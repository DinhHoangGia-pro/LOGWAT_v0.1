"""Random seed helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Required for CUDA's cuBLAS GEMM calls to be deterministic; must be set
    # before the CUDA context is created, so this has to happen as early as
    # possible (set_seed() is called first thing in train()).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[!] torch.use_deterministic_algorithms(True) failed (some op used by "
              f"GATv2Conv/scatter may not have a deterministic implementation on this "
              f"torch/torch_geometric version): {e}. Continuing WITHOUT full determinism -- "
              f"training may not be exactly reproducible across runs even with the same seed.")
