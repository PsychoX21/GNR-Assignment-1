import random
import deepnet_backend as backend

def seed_everything(seed):
    """
    Seed all random number generators for determinism.
    """
    random.seed(seed)
    backend.manual_seed(seed)
    # If numpy is used in the future, seed it here too
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
