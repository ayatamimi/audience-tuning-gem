
import os
from dataclasses import dataclass
from typing import Optional
import socket
from dotenv import load_dotenv
import yaml


@dataclass
class Config:
    # data
    data_root: str; train_subdir: str; val_subdir: str; input_size: int

    # training
    bs: int; epochs: int; lr: float; num_workers: int; beta: float; seed: int

    # model
    model_type: str; num_levels: int; codebook_size: int; codebook_dim: int; embed_dim: Optional[int]; 
    latent_channel: int; rotation_trick: bool; kmeans_init: bool; decay: float; learnable_codebook: bool; ema_update: bool; threshold_dead: Optional[int]

    # system
    world_size: int; local_rank: int; run_dir: str; torch_compile: bool


def _b(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return s.lower() in ("1", "true", "yes", "y", "on")

def _maybe_int(s: Optional[str]) -> Optional[int]:
    return int(s) if (s is not None and s != "" and s.lower() != "null") else None

def _load_env():
    # Load from .env or ENV_FILE if provided
    if load_dotenv is None:
        raise RuntimeError("Install python-dotenv to load ENV_FILE or .env")
    env_file = os.getenv("ENV_FILE")
    if env_file and os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        hostname = socket.gethostname().lower()
        if any(x in hostname for x in ("gpu01", "gpu02", "gpu03")):
            cluster_env = "envs/ini.env"
        elif "juwels" in hostname:
            cluster_env = "envs/juwels_booster.env"
        if os.path.exists(cluster_env):
            load_dotenv(dotenv_path=cluster_env, override=False)
        elif os.path.exists(".env"):  # last fallback
            load_dotenv(dotenv_path=".env", override=False)


def _load_config_file() -> dict:
    cfg_path = os.getenv("CONFIG_FILE")
    if not cfg_path:
        raise RuntimeError("CONFIG_FILE environment variable must point to a YAML config file.")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _get(d: dict, path: str, default=None):
    """dot-path getter: example: 'training.bs' returns d['training']['bs'] if present."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def get_config() -> Config:
    _load_env()
    f = _load_config_file()

    # ----- data -----
    data_root    = os.getenv("DATA_ROOT", _get(f, "data.data_root", "./data"))
    train_subdir = os.getenv("TRAIN_SUBDIR", _get(f, "data.train_subdir", "train"))
    val_subdir   = os.getenv("VAL_SUBDIR",   _get(f, "data.val_subdir",   "val"))
    input_size   = int(os.getenv("INPUT_SIZE", _get(f, "data.input_size", 256)))

    # ----- training -----
    bs          = int(os.getenv("BS",          _get(f, "training.bs", 4)))
    epochs      = int(os.getenv("EPOCHS",      _get(f, "training.epochs", 150)))
    lr          = float(os.getenv("LR",        _get(f, "training.lr", 3e-4)))
    num_workers = int(os.getenv("NUM_WORKERS", _get(f, "training.num_workers", 4)))
    beta        = float(os.getenv("BETA",      _get(f, "training.beta", 0.25)))
    seed        = int(os.getenv("SEED",        _get(f, "training.seed", 42)))

    # ----- model -----
    model_type       = os.getenv("MODEL_TYPE",    _get(f, "model.model_type", "UnconditionedHVQVAE"))
    num_levels       = int(os.getenv("NUM_LEVELS",_get(f, "model.num_levels", 1)))
    codebook_size    = int(os.getenv("CODEBOOK_SIZE", _get(f, "model.codebook_size", 512)))
    codebook_dim     = int(os.getenv("CODEBOOK_DIM",  _get(f, "model.codebook_dim", 1)))
    embed_dim        = _maybe_int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") is not None else _get(f, "model.embed_dim", None)
    embed_dim        = None if (embed_dim == "" or str(embed_dim).lower()=="none") else embed_dim
    latent_channel   = int(os.getenv("LATENT_CHANNEL", _get(f, "model.latent_channel", 128)))
    rotation_trick   = _b(os.getenv("ROTATION_TRICK"), _get(f, "model.rotation_trick", False))
    kmeans_init      = _b(os.getenv("KMEANS_INIT"),    _get(f, "model.kmeans_init", False))
    decay            = float(os.getenv("DECAY",        _get(f, "model.decay", 0.99)))
    learnable_codebook = _b(os.getenv("LEARNABLE_CODEBOOK"), _get(f, "model.learnable_codebook", False))
    ema_update       = _b(os.getenv("EMA_UPDATE"),     _get(f, "model.ema_update", True))
    threshold_dead   = _maybe_int(os.getenv("THRESHOLD_DEAD")) if os.getenv("THRESHOLD_DEAD") is not None else _get(f, "model.threshold_dead", None)

    # ----- system / launcher -----
    world_size    = int(os.getenv("WORLD_SIZE", "1"))
    local_rank    = int(os.getenv("LOCAL_RANK", "0"))
    run_dir       = os.getenv("RUN_DIR", _get(f, "system.run_dir", "./runs"))
    torch_compile = _b(os.getenv("TORCH_COMPILE"), _get(f, "system.torch_compile", False))

    return Config(
        data_root=data_root, train_subdir=train_subdir, val_subdir=val_subdir, input_size=input_size,
        bs=bs, epochs=epochs, lr=lr, num_workers=num_workers, beta=beta, seed=seed,
        model_type=model_type, num_levels=num_levels, codebook_size=codebook_size, codebook_dim=codebook_dim,
        embed_dim=embed_dim, latent_channel=latent_channel, rotation_trick=rotation_trick, kmeans_init=kmeans_init,
        decay=decay, learnable_codebook=learnable_codebook, ema_update=ema_update, threshold_dead=threshold_dead,
        world_size=world_size, local_rank=local_rank, run_dir=run_dir, torch_compile=torch_compile
    )
