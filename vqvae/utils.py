# vqvae/utils.py


import os
import random
import numpy as np
import torch

import vqvae.flat.flat_models as flat_models



def initialize_model(params):
    model_type = params["model_type"]
    try:
        cls = getattr(flat_models, model_type)  # e.g., UnconditionedHVQVAE
    except AttributeError:
        raise ValueError(f"Unknown model '{model_type}' in flat_models")

    return cls(
        in_channel=3,
        channel=params["latent_channel"],
        n_res_block=2,
        n_res_channel=params["latent_channel"] // 2,
        embed_dim=params.get("embed_dim", None),
        codebook_dim=params["codebook_dim"],
        n_embed=params["codebook_size"],
        decay=params["decay"],
        rotation_trick=params["rotation_trick"],
        kmeans_init=params["kmeans_init"],
        learnable_codebook=params["learnable_codebook"],
        ema_update=params["ema_update"],
        threshold_ema_dead_code=params.get("threshold_dead", None),
    )

def init_seeds(seed=None):
    seed = random.randint(0, 2147483647) if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
