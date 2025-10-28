#!/usr/bin/env python
# coding: utf-8

# # Set Up Modules

# In[1]:

# -*- coding: latin-1 -*-

import argparse, json, math, os, random, sys, re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils
from torchvision.models import resnet50, ResNet50_Weights
#from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
from torch.optim.lr_scheduler import CyclicLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from vqvae import FlatVQVAE
import torch.distributed as dist
import neptune.new as neptune
from transformers import DistilBertForMaskedLM, DistilBertConfig, get_linear_schedule_with_warmup
from PIL import Image, ImageDraw, ImageFont

from torch.utils.data import random_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def init_ddp_from_env():
    """
    Initialize torch.distributed only if launched with torchrun (env://).
    Safe no-op on single-GPU runs.
    """
    if not dist.is_available() or dist.is_initialized():
        return
    if ("RANK" in os.environ) or ("WORLD_SIZE" in os.environ) or ("LOCAL_RANK" in os.environ):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def is_primary() -> bool:
    return (not ddp_ready()) or dist.get_rank() == 0

def allreduce_sum_(t: torch.Tensor) -> torch.Tensor:
    if ddp_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def is_primary() -> bool:
    # True on single-GPU runs and on rank 0 when using DDP
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

import glob, os


def _latest_ckpt_in_distil_dir():
    ckpt_dir = "local/altamabp/checkpoint/distil"
    paths = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]  # <-- return a single string, not a list




def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """img_t: [C,H,W] in [-1,1] or [0,1] -> PIL RGB"""
    t = img_t.detach().cpu().float()
    if t.min() < 0:  # assume [-1,1]
        t = (t.clamp(-1, 1) + 1) / 2.0
    else:            # assume [0,1]
        t = t.clamp(0, 1)
    t = (t * 255.0).round().byte()
    npimg = t.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(npimg, mode="RGB")

def save_row_with_labels(tiles, labels, out_path, label_h=18, pad=2):
    """
    tiles: list of torch.Tensor [C,H,W], length N
    labels: list of str ('' to skip), length N
    Writes a single-row image with a label bar above each tile.
    """
    assert len(tiles) == len(labels)
    pil_tiles = [_tensor_to_pil(t) for t in tiles]
    w, h = pil_tiles[0].width, pil_tiles[0].height
    total_w = len(pil_tiles) * (w + pad) - pad
    total_h = h + label_h
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x = 0
    for im, text in zip(pil_tiles, labels):
        # label background strip
        if text:
            draw.rectangle([x, 0, x + w, label_h], fill=(20, 20, 20))
            tw, th = draw.textsize(text, font=font) # textsize is deprecated and will be removed in Pillow 10 (2023-07-01). Use textbbox or textlength instead.
            draw.text((x + (w - tw)//2, (label_h - th)//2), text, fill=(255, 255, 255), font=font)
        # paste image under label bar
        canvas.paste(im, (x, label_h))
        x += w + pad
    canvas.save(out_path)


def resolve_image_path(name, roots):
    """
    Return a real file path for 'name'.
    - If 'name' is already an absolute/existing path, return it.
    - Else try each root: join(root, name).
    - Else search recursively by basename under each root.
    """
    name = str(name)
    # already a full, existing path?
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    # try direct join with roots
    for r in roots:
        cand = os.path.join(r, name)
        if os.path.isfile(cand):
            return cand
    # fallback: search by basename
    base = os.path.basename(name)
    for r in roots:
        hits = glob.glob(os.path.join(r, "**", base), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not resolve image path for '{name}' under roots: {roots}")


def _latest_pt_in_dir(d, pattern="*.pt"):
    paths = glob.glob(os.path.join(d, pattern))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime)
    return paths[-1]

def _load_transformer_ckpt_into(model, optimizer, scheduler, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    # support both "model-only" and "full" checkpoints
    model_state = state.get("model", state)
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    # optional: restore opt/sched if present
    if isinstance(state, dict):
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
    # figure out last epoch (from state or filename)
    last_epoch = 0
    if isinstance(state, dict) and "epoch" in state:
        last_epoch = int(state["epoch"])
    else:
        m = re.search(r"epoch(\d+)", os.path.basename(ckpt_path))
        if m: last_epoch = int(m.group(1))
    return last_epoch

# # Train VQ_VAE

# > set up Logging 

# In[5]:


run = neptune.init_run(
    project="UTKFaces",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
    capture_stdout=False,
    capture_stderr=False,
    source_files=["pipeline_vqvae-transformer-classification_gpu01_correct_final-automated.py"]
)


# > Define a custom Dataset

# In[3]:


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        # Regex: first number, anything, last digit before .png
        self.pattern = re.compile(r'^(\d+)_.*_(\d)\.png$')

    def __len__(self):
        return len(self.image_files)

#
#    def __getitem__(self, idx):
#        img_name = self.image_files[idx]
#        img_path = os.path.join(self.image_dir, img_name)
#        image = Image.open(img_path).convert('RGB')
#        
#        # Extract labels using regex
#        m = self.pattern.match(img_name)
#        if not m:
#            raise ValueError(f"Filename {img_name} does not match expected pattern.")
#        first_number = m.group(1)
#        last_digit = m.group(2)
#        
#        # Pad first_number to at least 2 digits, then concatenate last_digit
#        # if len(first_number) == 1:
#        #     label = f"0{first_number}{last_digit}"  # e.g., 5 + 6 -> 056
#        # else:
#        label = int(f"{first_number}{last_digit}")   # e.g., 32 + 8 -> 328, 123 + 4 -> 1234
#
#        # Optionally, convert to int:
#        # label = int(label)
#
#        if self.transform:
#            image = self.transform(image)
#        return image, label
#

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Extract only the last digit from the filename
        m = self.pattern.match(img_name)
        if not m:
            raise ValueError(f"Filename {img_name} does not match expected pattern.")
        last_digit = m.group(2)          # group(2) is the last digit
        label = int(last_digit)          # convert it to int (0-9)

        if self.transform:
            image = self.transform(image)
        return image, label



# > Set Up train function

# In[23]:


def train(epoch, loader, model, optimizer, scheduler, device, val_loader=None):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from tqdm import tqdm

    # helper: true on single GPU or rank 0
    def _is_primary():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if _is_primary():
        loader = tqdm(loader, desc=f"[Train] Epoch {epoch+1}", unit="batch")

    criterion = nn.MSELoss(reduction="mean")
    latent_loss_weight = 0.35
    diversity_loss_weight = 0.0001

    model.train()

    # running (local only) for nice tqdm "avg mse"
    mse_sum_local = 0.0
    mse_n_local   = 0

    # epoch aggregates as tensors (DDP-safe)
    tr_total_sum_t = torch.tensor(0.0, device=device)
    tr_count_sum_t = torch.tensor(0.0, device=device)

    for i, (img, _) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        img = img.to(device, non_blocking=True)

        out, latent_loss, diversity_loss, codebook_usage = model(img)
        recon_loss  = criterion(out, img)         # mean over elements
        latent_loss = latent_loss.mean()
        if torch.is_tensor(diversity_loss) and diversity_loss.ndim > 0:
            diversity_loss = diversity_loss.mean()

        total_loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # for tqdm running avg
        bs = img.size(0)
        mse_sum_local += recon_loss.item() * bs
        mse_n_local   += bs

        # epoch totals (as tensors) for DDP reduction later
        tr_total_sum_t += total_loss.detach() * bs
        tr_count_sum_t += bs

        if _is_primary():
            lr = optimizer.param_groups[0]["lr"]
            loader.set_postfix(
                mse=f"{recon_loss.item():.5f}",
                latent=f"{latent_loss.item():.3f}",
                avg_mse=f"{(mse_sum_local/max(1,mse_n_local)):.5f}",
                lr=f"{lr:.5f}",
            )
            # keep your existing per-batch logs
            if "run" in globals() and run is not None:
                run["train/vqvae-mse"].log(recon_loss.item())
                run["train/vqvae-latent"].log(latent_loss.item())
                run["train/vqvae-epoch"].log(epoch + 1)
                run["train/vqvae-num_used_codebooks"].log(
                    float(codebook_usage.detach().item() if torch.is_tensor(codebook_usage) else codebook_usage)
                )

    # ---- DDP-safe global epoch average ----
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tr_total_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(tr_count_sum_t, op=dist.ReduceOp.SUM)
    train_loss_avg = (tr_total_sum_t / torch.clamp(tr_count_sum_t, min=1)).item()

    if _is_primary() and "run" in globals() and run is not None:
        run["train/vqvae-loss"].log(train_loss_avg, step=epoch + 1)
        run["train/vqvae-lr"].log(optimizer.param_groups[0]["lr"], step=epoch + 1)

    # (optional) do validation here and log validation/loss to compare curves
    if val_loader is not None:
        model.eval()
        va_total_sum_t = torch.tensor(0.0, device=device)
        va_count_sum_t = torch.tensor(0.0, device=device)
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}", unit="batch") if _is_primary() else val_loader
            for img, _ in vbar:
                img = img.to(device, non_blocking=True)
                out, latent_loss, diversity_loss, _ = model(img)
                recon_loss  = criterion(out, img)
                latent_loss = latent_loss.mean()
                if torch.is_tensor(diversity_loss) and diversity_loss.ndim > 0:
                    diversity_loss = diversity_loss.mean()
                total_loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
                bs = img.size(0)
                va_total_sum_t += total_loss.detach() * bs
                va_count_sum_t += bs
                if _is_primary():
                    vbar.set_postfix(loss=f"{total_loss.item():.5f}")

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(va_total_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(va_count_sum_t, op=dist.ReduceOp.SUM)
        val_loss_avg = (va_total_sum_t / torch.clamp(va_count_sum_t, min=1)).item()

        if _is_primary() and "run" in globals() and run is not None:
            run["validation/vqvae-loss"].log(val_loss_avg, step=epoch + 1)

        return train_loss_avg, val_loss_avg

    return train_loss_avg, None



# =============================================================================
# def train(epoch, loader, model, optimizer, scheduler, device):
#     if is_primary():
#         loader = tqdm(loader)
# 
#     criterion = nn.MSELoss()
# 
#     latent_loss_weight = 0.35
#     diversity_loss_weight = 0.0001
#     sample_size = 5
# 
#     mse_sum = 0
#     mse_n = 0
#     for i, (img, label) in enumerate(loader):
#         model.zero_grad()
# 
#         img = img.to(device)
#         out, latent_loss, diversity_loss, codebook_usage = model(img)
#         recon_loss = criterion(out, img)
#         latent_loss = latent_loss.mean()
#         loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
#         loss.backward()
# 
#         if scheduler is not None:
#             scheduler.step()
#         optimizer.step()
# 
#         part_mse_sum = recon_loss.item() * img.shape[0]
#         part_mse_n = img.shape[0]
#         comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
#         comm = dist.all_gather(comm)
# 
#         for part in comm:
#             mse_sum += part["mse_sum"]
#             mse_n += part["mse_n"]
# 
#         if is_primary():
#             lr = optimizer.param_groups[0]["lr"]
# 
#             loader.set_description(
#                 (
#                     f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
#                     f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
#                     f"lr: {lr:.5f}"
#                 )
#             )
#             run["train/mse"].log(recon_loss.item())
#             run["train/latent"].log(latent_loss.item())
#             run["train/epoch"].log(epoch + 1)
#             run["train/num_used_codebooks"].log(codebook_usage)
# =============================================================================


# > Parse essential arguments

# In[34]:


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)

port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
parser.add_argument("--save_path_models", default="/local/altamabp/checkpoint/vqvae/")
parser.add_argument("--size", type=int, default=80)
parser.add_argument("--epoch", type=int, default=100)#100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--sched", type=str)

### added
##--train_vqvae --train_classifier --train_transformer
parser.add_argument("--train_vqvae", action="store_true", help="Train VQ-VAE instead of loading checkpoint")
parser.add_argument("--train_classifier", action="store_true", help="Train classifier instead of loading checkpoint")
parser.add_argument("--train_transformer", action="store_true", help="Train transformer instead of loading checkpoint")

parser.add_argument("--classifier_ckpt", type=str, default="/local/altamabp/checkpoint/classifier/resnet50/weights_epoch100_fullTrain.pth", help="Path to classifier checkpoint")




parser.add_argument(
    "--ckpt_distil_combined",
    type=str,
    default="/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100_fullTrain.pt",
    help="Path to combined distillation checkpoint"
)
parser.add_argument(
    "--ckpt_vqvae",
    type=str,
    default="/local/altamabp/checkpoint/vqvae/model_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.pth",
    help="Path to trained VQ-VAE checkpoint"
)




args, unknown = parser.parse_known_args()

# args = parser.parse_args()
# args, unknown = parser.parse_known_args()


# In[35]:


torch.cuda.set_device(2)  # Use GPU 1 (if desired)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
args.distributed = ddp_ready() and (dist.get_world_size() > 1)


transform = transforms.Compose(
    [
        transforms.Resize((80,80)),
        transforms.Lambda(lambda x: x if isinstance(x, (bytes, bytearray)) else x),  # placeholder if needed
        transforms.ToTensor(),                             # PIL/uint8 → [0,1]
        transforms.Lambda(lambda t: t.clamp(0, 1)),  
    ]
)
# =============================================================================
# transform = transforms.Compose(
#     [
#         transforms.Resize((80,80)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
# =============================================================================


batchsize_modified=1
dataset = CustomImageDataset('/local/altamabp/UTKFace_dataset_subset_150000', transform=transform)#'/home/abghamtm/work/aya/image/original_sample_images', transform=transform)
data_loader = DataLoader(dataset, batch_size=16 // args.n_gpu, shuffle=False, num_workers=12)
print('len(dataset) : ',len(dataset))
print('len(dataset[0]): ', len(dataset[0]))
print('dataset[0][0].shape: ', dataset[0][0].shape)


# --- VQ-VAE train/val split ---
val_ratio = 0.10
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset_vq, val_dataset_vq = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader_vq = DataLoader(train_dataset_vq, batch_size=16 // args.n_gpu, shuffle=False,  num_workers=12)
val_loader_vq   = DataLoader(val_dataset_vq,   batch_size=16 // args.n_gpu, shuffle=False, num_workers=12)


def _build_name_to_idx(ds):
    # ImageFolder: ds.samples -> List[(path, class_idx)]
    if hasattr(ds, "samples"):
        return {os.path.basename(p): i for p, _ in ds.samples}
    # Custom dataset with .image_files list
    if hasattr(ds, "image_files"):
        return {os.path.basename(p): i for i, p in enumerate(ds.image_files)}
    # Fallback: slow linear scan (avoid if possible)
    return {}

train_name2idx = _build_name_to_idx(train_dataset_vq)
val_name2idx   = _build_name_to_idx(val_dataset_vq)

def load_from_vq_datasets_by_name(filename_basename):
    """Return tensor in the SAME normalization as VQ-VAE ([-1,1]) from train/val dataset."""
    if filename_basename in train_name2idx:
        img, _ = train_dataset_vq[train_name2idx[filename_basename]]  # tensor [C,H,W], already transformed
        return img
    if filename_basename in val_name2idx:
        img, _ = val_dataset_vq[val_name2idx[filename_basename]]
        return img
    raise FileNotFoundError(f"Image '{filename_basename}' not found in train_dataset_vq/val_dataset_vq")



model = FlatVQVAE().to(device)


if args.distributed:
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist.get_local_rank()],
        output_device=dist.get_local_rank(),
    )

optimizer = optim.Adam(model.parameters(), lr=args.lr)
run["train/lr"].log(args.lr)
scheduler = None
if args.sched == "cycle":
    scheduler = CyclicLR(
    optimizer, 
    base_lr=args.lr * 0.1, 
    max_lr=args.lr, 
    step_size_up=len(train_loader_vq) * args.epoch * 0.05, 
    mode="triangular",
    cycle_momentum=False)
    
    

    
def evaluate_vqvae(val_loader, model, device, run=None):
    import torch
    import torch.distributed as dist
    from tqdm import tqdm

    def _ddp_ready():
        return dist.is_available() and dist.is_initialized()

    def _is_primary():
        return (not _ddp_ready()) or dist.get_rank() == 0

    model.eval()

    sse_sum  = torch.tensor(0.0, device=device)  # sum of squared errors over ALL elements
    elem_sum = torch.tensor(0.0, device=device)  # total number of compared elements

    iterator = tqdm(val_loader, desc="[VQ-VAE] Valid", unit="batch") if _is_primary() else val_loader
    with torch.no_grad():
        for imgs, _ in iterator:
            imgs = imgs.to(device, non_blocking=True)
            recons, *_ = model(imgs)  # out, latent_loss, diversity_loss, codebook_usage

            diff = recons - imgs
            sse_sum  += (diff * diff).sum()
            elem_sum += torch.tensor(diff.numel(), device=device, dtype=torch.float32)

    # DDP: sum across all processes
    if _ddp_ready():
        dist.all_reduce(sse_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(elem_sum, op=dist.ReduceOp.SUM)

    # True per-element MSE
    val_mse = (sse_sum / torch.clamp(elem_sum, min=1.0)).item()

    if run is not None and _is_primary():
        run["vqvae/val_mse"].log(val_mse)

    return val_mse



if args.train_vqvae:
    print("Training VQ-VAE...")
    for i in range(args.epoch):
        train(i, train_loader_vq, model, optimizer, scheduler, device)
        if is_primary():
            model_path = os.path.join(args.save_path_models, f"model_epoch{i+1}_flat_vqvae80x80_144x456codebook_fullTrain.pth")
            torch.save(model.state_dict(), model_path)

    # ---- One-shot VQ-VAE validation (MSE) ----
#    if dist.is_primary():
        v_mse = evaluate_vqvae(val_loader_vq, model, device, run)
        print(f"[VQ-VAE] Final validation MSE: {v_mse:.6f}")

else:
    print(f"Loading VQ-VAE from {args.ckpt_vqvae}")
    model.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model.eval()
    


# =============================================================================
# x=0
# for i in range(args.epoch):
#     train(i, data_loader, model, optimizer, scheduler, device)
#     x=i
#     if dist.is_primary():
#         model_path = os.path.join(args.save_path_models, f"model_epoch{i+1}_flat_vqvae80x80_144x456codebook.pth")
#         torch.save(model.state_dict(), model_path)
# 
# =============================================================================

# # Reconstruction Step
# 
# - Reconstruct Images
# - Save Codebook
# - Save Latent Space - Indices
# - Save labels properly

# In[36]:


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)

port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
parser.add_argument("--save_path_models", default="/local/altamabp/checkpoint/vqvae/")
parser.add_argument("--save_path_imgs", default="/local/altamabp/image/reconstruction_fullTrain/")
parser.add_argument("--size", type=int, default=80)
parser.add_argument("--epoch", type=int, default=100)#100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--sched", type=str)
parser.add_argument('--ckpt_vqvae', type=str, default="/local/altamabp/checkpoint/vqvae/model_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.pth")

### added
parser.add_argument(
    "--ckpt_distil_combined",
    type=str,
    default="/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100_fullTrain.pt",
    help="Path to combined distillation checkpoint"
)
parser.add_argument("--train_vqvae", action="store_true", help="Train VQ-VAE instead of loading checkpoint")
parser.add_argument("--train_classifier", action="store_true", help="Train classifier instead of loading checkpoint")
parser.add_argument("--train_transformer", action="store_true", help="Train transformer instead of loading checkpoint")

parser.add_argument("--classifier_ckpt", type=str, default="/local/altamabp/checkpoint/classifier/resnet50/weights_epoch100_fullTrain.pth", help="Path to classifier checkpoint")


args, unknown = parser.parse_known_args()


# In[37]:

    


torch.cuda.set_device(2)  # Use GPU 1 (if desired)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize DDP if launched with torchrun; otherwise stays single-process
init_ddp_from_env()

# SAFE distributed flag (won’t throw if not initialized)
args.distributed = ddp_ready() and (dist.get_world_size() > 1)


# =============================================================================
# transform = transforms.Compose(
#     [
#         transforms.Resize((80,80)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
# 
# =============================================================================
dataset = CustomImageDataset('/local/altamabp/UTKFace_dataset_subset_150000', transform=transform)
data_loader = DataLoader(dataset, batch_size=16 // args.n_gpu, shuffle=False, num_workers=12)

model_vqvae = FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()
epoch = args.epoch

if args.distributed:
    model_vqvae = nn.parallel.DistributedDataParallel(
        model_vqvae,
        device_ids=[dist.get_local_rank()],
        output_device=dist.get_local_rank(),
    )
    
    
    

os.makedirs(args.save_path_models, exist_ok=True)
if is_primary():
    if args.train_vqvae:
        model_vqvae.eval()
        all_indices = []
        all_quantizes = []
        all_labels = []
        all_img_names = []  # <-- NEW: keep filenames aligned with quant/indices
    
        with torch.no_grad():
            for j, (images, labels) in tqdm(
                enumerate(data_loader),
                desc='Recall trained model to save codebook indices',
                leave=False
            ):
                # Get this batch's file paths IN THE SAME ORDER as 'images' in this batch.
                # This slicing assumes no shuffle; if shuffled, return path from __getitem__ instead.
                batch_paths = data_loader.dataset.image_files[
                    j * data_loader.batch_size : (j + 1) * data_loader.batch_size
                ]
                # If you prefer basenames, use:
                # batch_names = [os.path.basename(p) for p in batch_paths]
                # Otherwise, keep full paths (recommended to avoid collisions):
                batch_names = list(batch_paths)
    
                images = images.float().to(device)
                quant_b, _, id_b, _, _ = model_vqvae.encode(images)
                outputs = model_vqvae.decode(quant_b)
    
                # Record filenames for this batch (must be same length as batch size)
                all_img_names.extend(batch_names)  # <-- NEW
    
                # (your optional visualization block)
                for idx, (label, out, img_path) in enumerate(zip(labels, outputs, batch_names)):
                    if idx % 10000 == 0 or idx == 0:
                        save_file = os.path.join(
                            args.save_path_imgs,
                            f"{os.path.splitext(os.path.basename(img_path))[0]}_{label.item()}.png"
                        )
                        orig = images[idx]
                        grid = torch.stack([orig.detach().cpu(), out.detach().cpu()], dim=0)
                        grid_01 = (grid.clamp(-1, 1) + 1) / 2.0
                        utils.save_image(grid_01, save_file, nrow=2)
    
                all_indices.append(id_b.cpu())
                all_quantizes.append(quant_b.cpu())
                all_labels.extend(labels.cpu().numpy())
    
        # Concatenate & save
        indices_tensor   = torch.cat(all_indices, dim=0)
        quantizes_tensor = torch.cat(all_quantizes, dim=0)
        all_labels       = np.array(all_labels)
    
        # sanity check: lengths must match
        assert indices_tensor.shape[0] == quantizes_tensor.shape[0] == len(all_img_names), \
            f"Length mismatch: indices={indices_tensor.shape[0]}, quant={quantizes_tensor.shape[0]}, names={len(all_img_names)}"
    
        indices_path   = os.path.join(args.save_path_models, f"indices_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")
        quantized_path = os.path.join(args.save_path_models, f"quantized_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")
        labels_path    = os.path.join(args.save_path_models, f"labels_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")
        names_path     = os.path.join(args.save_path_models, f"filenames_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")  # <-- NEW
    
        np.save(indices_path,   indices_tensor.numpy())
        np.save(quantized_path, quantizes_tensor.numpy())
        np.save(labels_path,    all_labels)
        np.save(names_path,     np.array(all_img_names, dtype=object))  # <-- NEW


# =============================================================================
# # Save the final model
# os.makedirs(args.save_path_models, exist_ok=True)
# if is_primary():
#     model_vqvae.eval()
#     all_indices = []
#     all_quantizes = []
#     all_labels = []
#     with torch.no_grad():
#         for j, (images, labels) in tqdm(enumerate(data_loader), desc='Recall trained model to save codebook indices', leave=False): # Ignore the label in DataLoader
#             # print(labels)
#             
#             batch_indices = data_loader.dataset.image_files[j*data_loader.batch_size : (j+1)*data_loader.batch_size]
# 
# 
#             images = images.float().to(device)
#             quant_b, _, id_b, _, _ = model_vqvae.encode(images)
#             outputs = model_vqvae.decode(quant_b)
#             
#             for idx, (label, out, img_name) in enumerate(zip(labels, outputs, batch_indices)):
#                 if idx % 10000 == 0 or idx == 0:
#                     save_file = os.path.join(
#                         args.save_path_imgs,
#                         f"{os.path.splitext(img_name)[0]}_{label.item()}.png"
#                     )
#             
#                     # original input matching this output
#                     orig = images[idx]  # shape [C,H,W], same normalization as training
#             
#                     # stack: [2, C, H, W] = [original, reconstruction]
#                     grid = torch.stack([
#                         orig.detach().cpu(),
#                         out.detach().cpu()
#                     ], dim=0)
#             
#                     utils.save_image(
#                         grid,
#                         save_file,
#                         nrow=2,                # two tiles per row (orig | recon)
#                         normalize=True,        # normalize for viewing
#                         value_range=(-1, 1)          # because your tensors are in [-1, 1]
#                     )
#             
# # =============================================================================
# #             for idx, (label, out, img_name) in enumerate(zip(labels, outputs, batch_indices)):
# #                 # print(label)
# # 
# #                 if idx % 10000 == 0 or idx == 0:
# #                     save_file = os.path.join(args.save_path_imgs, f"{os.path.splitext(img_name)[0]}_{label.item()}.png")
# #                     # print(save_file)
# #                     utils.save_image(
# #                         torch.cat([out.unsqueeze(0)], 0),
# #                         save_file,
# #                         nrow=2,
# #                         normalize=True,
# #                         #range=(-1, 1),
# #                     )
# # =============================================================================
# 
#             all_indices.append(id_b.cpu())
#             all_quantizes.append(quant_b.cpu())
#             all_labels.extend(labels.cpu().numpy())
#             
# 
#     # Concatenate all indices into a single tensor and save it
#     indices_tensor = torch.cat(all_indices, dim=0)
#     quantizes_tensor = torch.cat(all_quantizes, dim=0)
#     all_labels = np.array(all_labels)
# 
#     indices_path = os.path.join(args.save_path_models, f"indices_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")
#     quantized_path = os.path.join(args.save_path_models, f"quantized_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy")
#     np.save(os.path.join(args.save_path_models, f'labels_epoch{epoch}_flat_vqvae80x80_144x456codebook_fullTrain.npy'), all_labels)
# 
#     np.save(indices_path, indices_tensor.numpy())
#     np.save(quantized_path, quantizes_tensor.numpy())
# =============================================================================


# # Train a Classifier

# In[4]:


class ReconstructedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


# In[ ]:



def decode_quantizes(model, quantizes, batch_size=32, device="cuda"):
    """
    Decode quantized latent codes into images in batches to avoid CUDA OOM.
    Args:
        model: trained VQ-VAE model
        quantizes: numpy array [N, C, H, W] or tensor [N, C, H, W]
        batch_size: how many samples to decode per batch
        device: "cuda" or "cpu"
    Returns:
        images: concatenated tensor [N, C, H, W]
    """
    # 🔹 Debug info once
    if isinstance(quantizes, np.ndarray):
        print(f"[decode_quantizes] quantizes is numpy array, shape={quantizes.shape}, dtype={quantizes.dtype}")
    elif torch.is_tensor(quantizes):
        print(f"[decode_quantizes] quantizes is torch tensor, shape={tuple(quantizes.shape)}, dtype={quantizes.dtype}")
    else:
        raise TypeError(f"quantizes must be numpy.ndarray or torch.Tensor, got {type(quantizes)}")

    model.eval()
    images = []
    with torch.no_grad():
        for i in range(0, len(quantizes), batch_size):
            batch = quantizes[i:i+batch_size]
            if isinstance(batch, np.ndarray):  # convert numpy → tensor if needed
                batch = torch.from_numpy(batch).float()
            batch = batch.to(device, non_blocking=True)

            out = model.decode(batch)
            images.append(out.cpu())

            del batch, out
            torch.cuda.empty_cache()

    return torch.cat(images, dim=0)


# =============================================================================
# def decode_quantizes(model, quantizes, batch_size=32, device="cuda"):
#     """
#     Decode quantized latent codes into images in batches to avoid CUDA OOM.
#     Args:
#         model: trained VQ-VAE model
#         quantizes: tensor [N, C, H, W] or [N, T, D]
#         batch_size: how many samples to decode per batch
#         device: "cuda" or "cpu"
#     Returns:
#         images: concatenated tensor [N, C, H, W]
#     """
#     model.eval()
#     images = []
#     with torch.no_grad():
#         for i in range(0, len(quantizes), batch_size):
#             batch = quantizes[i:i+batch_size].to(device, non_blocking=True)
#             out = model.decode(batch)
#             images.append(out.cpu())
#             # free up GPU memory
#             del batch, out
#             torch.cuda.empty_cache()
#     return torch.cat(images, dim=0)
# =============================================================================

# =============================================================================
# def decode_quantizes(model, quantizes):
#     quantizes = torch.tensor(quantizes).float().to(device)
#     with torch.no_grad():
#         images = model.decode(quantizes)
#     return images
# =============================================================================

ckpt_vqvae = "/local/altamabp/checkpoint/vqvae/model_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.pth"
torch.cuda.set_device(2) 
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

quantizes = np.load('/local/altamabp/checkpoint/vqvae/quantized_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
labels = np.load('/local/altamabp/checkpoint/vqvae/labels_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
labels = torch.from_numpy(labels)

model_vqvae = FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=device))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()
#reconstructed_images = decode_quantizes(model_vqvae, quantizes)
reconstructed_images = decode_quantizes(model_vqvae, quantizes, batch_size=32, device=device)



dataset = ReconstructedDataset(reconstructed_images, labels)

# # Split the indices in a stratified manner
# train_indices, test_indices = train_test_split(
#     np.arange(len(labels)),
#     test_size=0.2,
#     stratify=labels,
#     random_state=42
# )
# # Create subsets of the dataset
# train_dataset = Subset(dataset, train_indices)
# test_dataset = Subset(dataset, test_indices)

data_loader_reconstructions = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0) # It only works when you set the num_worker to 0
for inputs, labels in data_loader_reconstructions:
    print('reconstructions inputs.shape: ', inputs.shape)
    print('reconstructions labels.shape: ', labels.shape)
    break

# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12)


from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

# Transforms (same as your pipeline)
# =============================================================================
# transform = transforms.Compose([
#     transforms.Resize((80,80)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
# ])
# =============================================================================

# Dataset + DataLoader
train_dataset = CustomImageDataset('/local/altamabp/UTKFace_dataset_subset_150000', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


val_dataset = datasets.ImageFolder("UTKFace_dataset_test_structured", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



#validation loop outside training
from tqdm import tqdm

# --------------------------
# Classifier (ResNet50)
# --------------------------

classifier = resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)  
classifier.to(device)

nn.init.normal_(classifier.fc.weight, mean=0.0, std=0.01)
nn.init.zeros_(classifier.fc.bias)
# =============================================================================
# classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# classifier.fc = nn.Linear(classifier.fc.in_features, 10)
# classifier.to(device)
# =============================================================================



def _epoch_loop(loader, model, criterion, device, optimizer=None, desc=""):
    """
    One pass over `loader`. If `optimizer` is provided -> training, else validation.
    Returns (loss_avg, acc_pct).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, unit="batch")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        # accumulate using sample-weighted loss (robust to last batch size)
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        _, predicted = torch.max(outputs, 1)
        total += bs
        correct += (predicted == labels).sum().item()

        # live progress shows running averages
        run_loss = loss_sum / max(1, total)
        run_acc = 100.0 * correct / max(1, total)
        pbar.set_postfix({"loss": f"{run_loss:.4f}", "acc": f"{run_acc:.2f}%"})

    epoch_loss = loss_sum / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    return epoch_loss, epoch_acc

if args.train_classifier:
    print("Training classifier...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 100  # change as needed
    for epoch in range(num_epochs):
        # ---- Train (one pass) ----
        train_loss, train_acc = _epoch_loop(
            train_loader, classifier, criterion, device,
            optimizer=optimizer,
            desc=f"[Train] Epoch {epoch+1}/{num_epochs}"
        )

        # ---- Validate (one pass) ----
        val_loss, val_acc = _epoch_loop(
            val_loader, classifier, criterion, device,
            optimizer=None,
            desc=f"[Val]   Epoch {epoch+1}/{num_epochs}"
        )

        # ---- Logging (per epoch) ----
        if "run" in globals() and run is not None:
            try:
                run["train/classifier-loss"].log(train_loss, step=epoch + 1)
                run["train/classifier-acc"].log(train_acc, step=epoch + 1)
                run["train/classifier-lr"].log(optimizer.param_groups[0]["lr"], step=epoch + 1)

                run["validation/classifier-loss"].log(val_loss, step=epoch + 1)
                run["validation/classifier-acc"].log(val_acc, step=epoch + 1)
            except Exception:
                pass

        print(
            f"[Classifier] Epoch {epoch+1}/{num_epochs} -- "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% "
            f"-- Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

        # Save checkpoint (optional)
        torch.save(
            classifier.state_dict(),
            f"/local/altamabp/checkpoint/classifier/resnet50/weights_epoch{str(epoch+1).zfill(2)}_fullTrain.pth"
        )

else:
    print(f"Loading classifier from {args.classifier_ckpt}")
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()



# =============================================================================
#validation loop inside training
# from tqdm import tqdm
# 
# # --------------------------
# # Classifier (ResNet50)
# # --------------------------
# classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# classifier.fc = nn.Linear(classifier.fc.in_features, 10)
# classifier.to(device)
# 
# if args.train_classifier:
#     print("Training classifier...")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
# 
#     num_epochs = 1  # change as needed
#     for epoch in range(num_epochs):
#         # ---- Training ----
#         classifier.train()
#         running_loss, correct, total = 0.0, 0, 0
# 
#         pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", unit="batch")
#         for inputs, labels in pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
# 
#             optimizer.zero_grad()
#             outputs = classifier(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
# 
#             acc = 100 * correct / total
#             pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2f}%"})
# 
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
# 
#         # ---- Validation ----
#         classifier.eval()
#         val_loss, val_correct, val_total = 0.0, 0, 0
# 
#         vbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}/{num_epochs}", unit="batch")
#         with torch.no_grad():
#             for inputs, labels in vbar:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = classifier(inputs)
#                 loss = criterion(outputs, labels)
# 
#                 val_loss += loss.item()
#                 run["validation/loss"].log(val_loss)
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
# 
#                 val_acc = 100 * val_correct / val_total
#                 vbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{val_acc:.2f}%"})
# 
#         val_loss /= len(val_loader)
#         val_acc = 100 * val_correct / val_total
# 
#         print(f"[Classifier] Epoch {epoch+1}/{num_epochs} "
#               f"-- Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% "
#               f"-- Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
# 
#         # Save checkpoint
#         torch.save(
#             classifier.state_dict(),
#             f"/local/altamabp/checkpoint/classifier/resnet50/weights_epoch{str(epoch+1).zfill(2)}.pth"
#         )
# 
# else:
#     print(f"Loading classifier from {args.classifier_ckpt}")
#     classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
#     classifier.eval()
# 
# =============================================================================





# =============================================================================
# # --------------------------
# # Classifier (ResNet50)
# # --------------------------
# classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# classifier.fc = nn.Linear(classifier.fc.in_features, 10)
# classifier.to(device)
# 
# if args.train_classifier:
#     print("Training classifier...")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
# 
#     for epoch in range(1):
#         running_loss, correct, total = 0.0, 0, 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = classifier(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
# 
#         acc = 100 * correct / total
#         print(f"[Classifier] Epoch {epoch+1}, Loss={running_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
# 
#         # Save checkpoint
#         torch.save(
#             classifier.state_dict(),
#             f"/local/altamabp/checkpoint/classifier/resnet50/weights_epoch{str(epoch+1).zfill(2)}.pth"
#         )
# else:
#     print(f"Loading classifier from {args.classifier_ckpt}")
#     classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
#     classifier.eval()
# =============================================================================


# =============================================================================
# # Training
# for epoch in range(100):  
#     model.train()
#     running_loss, correct, total = 0.0, 0, 0
#     
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
# 
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
# 
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     
#     print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")
# 
# 
#     torch.save(model.state_dict(), f"/local/altamabp/checkpoint/classifier/resnet50/weights_epoch{str(epoch + 1).zfill(2)}.pth")
# 
# =============================================================================

# =============================================================================
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 
# # Training loop
# num_epochs = 30#1#30  # Adjust as needed
# 
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for inputs, labels in tqdm(data_loader):
#         inputs = preprocess(inputs)
#         inputs, labels = inputs.to(device), labels.to(device)
# 
#         # Zero the parameter gradients
#         optimizer.zero_grad()
# 
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
# 
#         # Backward pass
#         loss.backward()
#         optimizer.step()
# 
#         running_loss += loss.item()
# 
#         # Accuracy calculation
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# 
#     # Calculate average loss and accuracy
#     epoch_loss = running_loss / len(data_loader)
#     epoch_accuracy = 100 * correct / total
# 
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
#     torch.save(model.state_dict(), f"/local/altamabp/checkpoint/classifier/resnet50/weights_epoch{str(epoch + 1).zfill(2)}.pth")
# 
# =============================================================================
# # Define classifier and load saved model(weights)
# classifier = resnet50(pretrained=False)
# models_list = os.listdir("/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/")
# models_list.sort()
# for model_name in models_list:
#     print(model_name)
#     classifier.load_state_dict(torch.load(os.path.join("/home/abghamtm/work/aya/checkpoint/classifier/resnet50/",model_name)))
#     classifier.to(device)
#     classifier.eval()  # Set model to evaluation mode
#     correct = 0
#     total = 0
#     total_loss = 0

#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader):
#             inputs = preprocess(inputs)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = classifier(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()

#     accuracy = correct / total
#     print(f'Accuracy: {accuracy * 100:.2f}%')
#     print(f'loss: {total_loss:.2f}%')


# # Train Transformer

# In[9]:

# =============================================================================
# 
# run = neptune.init_run(
#     project="UTKFaces",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
#     capture_stdout = False,
#     capture_stderr = False
# #    with_id="VQVAET-6"
# )
# =============================================================================


# In[10]:

    
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------
# Trasformer Keras-like "fit" for PyTorch
# ----------------------------
def fit_distil_with_embeds(
    model: torch.nn.Module,
    train_inputs_embeds: torch.Tensor,   # shape: (N_train, T, H)
    train_labels: torch.Tensor,          # shape: (N_train, T), -100 where ignored
    val_inputs_embeds: torch.Tensor,     # shape: (N_val, T, H)
    val_labels: torch.Tensor,            # shape: (N_val, T), -100 where ignored
    lr: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    n_warmup_epochs: int = 1,
    ckpt_dir: str = "mdl_checkpoints_800",
    resume_weights: str = "distilBERT_weights_new_800.pt",
    device: torch.device = None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    mask = (torch.rand((val_inputs_embeds.shape[1],), device=val_inputs_embeds.device) < 0.75) #added

    # Datasets & loaders
    train_ds = TensorDataset(train_inputs_embeds, train_labels)
    val_ds   = TensorDataset(val_inputs_embeds,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=False)

    # Move model first, then build optimizer/scheduler
    model.to(device)

    # Optimizer (AdamW like TF AdamWeightDecay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear schedule with warmup (epoch-based like your TF schedule)
    total_steps  = n_epochs * len(train_loader)
    warmup_steps = n_warmup_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Resume (optional)
# =============================================================================
#     if Path(resume_weights).is_file():
#         print(f"[INFO] Resuming weights: {resume_weights}")
#         state = torch.load(resume_weights, map_location=device)
#         model.load_state_dict(state, strict=False)
# =============================================================================

    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0

        for batch in tqdm (train_loader):
            inputs, labels = [t.to(device) for t in batch]  # shapes: (B,T,H) and (B,T)
            # HuggingFace computes MLM loss if labels are provided (CrossEntropy with ignore_index=-100)
            out = model(inputs_embeds=inputs, labels=labels)
            loss = out.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        run["train/transformer-loss"].log(train_loss)

        # ---- validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = [t.to(device) for t in batch]
                out = model(inputs_embeds=inputs, labels=labels)
                val_running += out.loss.item()
        val_loss = val_running / max(1, len(val_loader))
        run["validation/loss"].log(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save every epoch (like your ModelCheckpoint) and best
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_new.pt")
        torch.save(model.state_dict(), epoch_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "distilBERT_weights_new_800.pt")  # best weights
            
        ###added    
# =============================================================================
#         outputs = model(inputs_embeds=val_inputs_embeds[:3], output_hidden_states=True)
# 
#         # 1) argmax -> tokens
#         pred_tokens = torch.argmax(outputs.logits, dim=2)[0].to(device).long()
# 
#         # 2) clamp to valid vocab (defensive)
#         vmax = model.config.vocab_size - 1
#         pred_tokens.clamp_(0, vmax)
# 
#         # 3) copy original index grid, fill only masked positions, ensure long dtype
#         confidence_based_recons_index = val_labels[:3].clone().long()
#         confidence_based_recons_index[mask] = pred_tokens[mask]
#         
#         # Confirm shapes and dtypes before decoding
#         assert confidence_based_recons_index.dtype == torch.long
#         assert confidence_based_recons_index.shape == (n_token,)
# 
#         # 4) decode using LONG index grid of shape [1, H, W]
#         distil_out = model_vqvae.decode_code(
#             confidence_based_recons_index.reshape(1, length, length).long()
#         )
#         plot_distil_out(distil_out,epoch)
# =============================================================================
        ###


    print(f"[DONE] Best val loss: {best_val:.4f}")
    return model




# =============================================================================
# before movinf to keras like training
# def train(epoch, loader, model, optimizer, scheduler, device, val_loader=None):
#     if is_primary():
#         loader = tqdm(loader)
#     model.train()
#     
#     loss = 0
#     i=0
#     for i, (input, label) in enumerate(loader):
#         
#         model.zero_grad()
# 
#         # ensure devices
#         input = input.to(device)
#         label = label.to(device)
#         
#         # ensure model is on cuda:3 (and wrapped correctly)
#         base = model.module if isinstance(model, nn.DataParallel) else model
#         assert next(base.parameters()).is_cuda and next(base.parameters()).device.index == 3
#         
#         
#         outputs = model(inputs_embeds = input,labels =label)
#         index = torch.argmax(outputs.logits, dim=2)
# 
#         loss, logits = outputs[:2]
#         
#         loss = loss.mean()
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         if scheduler is not None:
#             scheduler.step()
#         
# 
#         if is_primary():
#             lr = optimizer.param_groups[0]["lr"]
# 
#             loader.set_description(
#                 (
#                     f"epoch: {epoch + 1}; loss: {loss:.5f}; "
#                     f"lr: {lr:.5f}"
#                 )
#             )
#         run["train/transformer-loss"].log(loss)
#         run["train/transformer-lr"].log(lr)
# 
# 
# # =============================================================================
# #         ##validation
# #         if val_loader is not None:
# #             if dist.is_primary():
# #                 val_loader = tqdm(val_loader)
# #                 model.eval()
# #             average_loss = 0
# #             val_loss = 0
# #             i=0    
# #             j=0
# #             for i, (input, label) in enumerate(val_loader):
# #             
# #                 if(i%500 ==0):
# #                     j = j+1
# #                     model.zero_grad()
# #                     
# #                     input = input.to(device)
# #                     label = label.to(device)
# #                     
# #                     i = i+1
# #                     outputs = model(inputs_embeds = input,labels =label)
# #                     val_loss, _ = outputs[:2]
# #                     val_loss = val_loss.mean()
# #                     run["validation/loss"].log(val_loss)
# #                     average_loss += val_loss
# #     
# #                     val_loader.set_description(
# #                             (
# #                                 f"Validation loss: {val_loss:.5f} "
# #                             )
# #                         )
# #             average_loss = average_loss/ j
# #             run["validation/average_loss_per_epoch"].log(average_loss)
# #             return average_loss
# # =============================================================================
# =============================================================================
        



# In[11]:


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)

port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

#parser.add_argument("--size", type=int, default=80)
parser.add_argument("--epoch", type=int, default=800)#100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--sched", type=str, default="linearW")


### added
parser.add_argument(
    "--ckpt_distil_combined",
    type=str,
    default="/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100_fullTrain.pt",
    help="Path to combined distillation checkpoint"
)
parser.add_argument(
    "--ckpt_vqvae",
    type=str,
    default="/local/altamabp/checkpoint/vqvae/model_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.pth",
    help="Path to trained VQ-VAE checkpoint"
)

parser.add_argument("--train_vqvae", action="store_true", help="Train VQ-VAE instead of loading checkpoint")
parser.add_argument("--train_classifier", action="store_true", help="Train classifier instead of loading checkpoint")
parser.add_argument("--train_transformer", action="store_true", help="Train transformer instead of loading checkpoint")

parser.add_argument("--classifier_ckpt", type=str, default="/local/altamabp/checkpoint/classifier/resnet50/weights_epoch100_fullTrain.pth", help="Path to classifier checkpoint")


parser.add_argument("--continue_train_transformer", action="store_true",
                    help="Resume transformer training for --epoch more epochs.")
parser.add_argument("--resume_transformer_path", type=str, default="",
                    help="Optional explicit transformer checkpoint path (.pt). If empty, picks latest from /local/altamabp/checkpoint/distil/")


args, unknown = parser.parse_known_args()


# In[ ]:


torch.cuda.set_device(2)  # Set default CUDA device to 3
torch.cuda.empty_cache()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
args.distributed = ddp_ready() and (dist.get_world_size() > 1)

#### train set###
train_labels = np.load('/local/altamabp/checkpoint/vqvae/labels_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')

train_indices = np.load('/local/altamabp/checkpoint/vqvae/indices_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
n, h, w = train_indices.shape
train_indices = train_indices.reshape(n, h * w)

train_quantizes = np.load('/local/altamabp/checkpoint/vqvae/quantized_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
n, c, h, w = train_quantizes.shape
train_quantizes = train_quantizes.transpose(0, 2, 3, 1)
train_quantizes = train_quantizes.reshape(n, h * w, c)

n_train_samples = train_quantizes.shape[0]
d_embed_vec = train_quantizes.shape[2]
n_tokens = train_quantizes.shape[1]

mask_token = 0
#mask_token_label = -100
mask_perc = 0.75
mask_train = np.random.default_rng().choice([True, False], size=(n_train_samples, n_tokens), p=[mask_perc, 1 - mask_perc])
train_quantizes = train_quantizes.reshape((n_train_samples, n_tokens, d_embed_vec))
train_indices = train_indices.reshape((n_train_samples, n_tokens))
train_quantizes[mask_train] = mask_token
masked_train_indices = np.copy(train_indices)
train_quantizes[mask_train] = mask_token

train_indices_label = np.copy(train_indices)
#train_indices_label[~mask_train] = mask_token_label
masked_train_quantizes = torch.from_numpy(train_quantizes)
masked_train_indices = torch.from_numpy(masked_train_indices)
train_indices_label = torch.from_numpy(train_indices_label)

indices = set(train_indices.flatten())
indices = sorted(indices)
vocab_size = indices[-1] + 1 

print('indices: ', indices)
print('indices max: ', max(indices))
print('indices min: ', min(indices))

# added during UTKFace debugging
train_indices_label = train_indices_label.to(device).long()  # enforce int64


###begin added
N, T = train_indices.shape
num_classes = 10

idx_t = torch.as_tensor(train_labels, dtype=torch.long, device=masked_train_quantizes.device)  # (N,)

N, T, _ = masked_train_quantizes.shape
idx_t_exp = idx_t.unsqueeze(1).expand(N, T)                         # (N, T)

one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=num_classes).to(dtype=masked_train_quantizes.dtype)  # (N,T,C)

masked_exp_train_quantizes = torch.cat([masked_train_quantizes, one_hot_labels], dim=2)


print('masked_exp_train_quantizes.shape: ',masked_exp_train_quantizes.shape)  # torch.Size([160000, 400, 74])
# masked_exp_train_quantizes: torch.FloatTensor (N, T, 74)
# mask_train: numpy bool array (N, T)  True = masked

device = masked_exp_train_quantizes.device
dtype  = masked_exp_train_quantizes.dtype

# 1) NumPy -> Torch, cast to float (1.0 masked, 0.0 unmasked)
mask_feat = torch.as_tensor(mask_train, device=device).to(dtype)   # (N, T)

# 2) Add feature axis
mask_feat = mask_feat.unsqueeze(-1)                                 # (N, T, 1)

# 3) Concatenate
masked_exp_mask_feat_train_quantizes = torch.cat([masked_exp_train_quantizes, mask_feat], dim=-1)  # (N, T, 75)

print('q_norm_plus_mask.shape: ',masked_exp_mask_feat_train_quantizes.shape) 
### end added


# --- Transformer train/val split (10%) ---
n_total = masked_exp_mask_feat_train_quantizes.shape[0]
idx_all = np.arange(n_total)
tr_idx, val_idx = train_test_split(
    idx_all, test_size=0.10, random_state=42, shuffle=False
)

x_tr  = masked_exp_mask_feat_train_quantizes[tr_idx]
y_tr  = train_indices_label[tr_idx]
x_val = masked_exp_mask_feat_train_quantizes[val_idx]
y_val = train_indices_label[val_idx]

train_data = ReconstructedDataset(x_tr,  y_tr)
val_data   = ReconstructedDataset(x_val, y_val)

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)   # , num_workers=...
val_dataloader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)  # , num_workers=...




# =============================================================================
# train_data = ReconstructedDataset(train_quantizes, train_indices_label)
# train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) #, num_workers=12
# =============================================================================

cfg = DistilBertConfig(
    vocab_size=vocab_size,
    hidden_size=d_embed_vec+11,
    sinusoidal_pos_embds=False,
    n_layers=6,
    n_heads=5,#4,
    max_position_embeddings=n_tokens
)

model = DistilBertForMaskedLM(cfg)
# model.load_state_dict(torch.load(args.ckpt_distil))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# # DataParallel: use device_ids=[3] because you want to use GPU 3
# model = torch.nn.DataParallel(model, device_ids=[3], output_device=3)
# model = model.to(device)
# 
# if args.distributed:
#     model = nn.parallel.DistributedDataParallel(
#         model,
#         device_ids=[3],
#         output_device=3,
#     )
# =============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
scheduler = None
if args.sched == "linearW":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        pct_start=0.001,
        steps_per_epoch=len(train_dataloader),
        epochs=args.epoch,
        anneal_strategy='linear'
    )


def evaluate_transformer(model, val_loader, device, run=None):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="[Transformer] Valid", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs_embeds=inputs, labels=labels)
            loss = outputs.loss.mean()
            bs = inputs.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    val_loss = total_loss / max(1, total_n)
    #if run is not None and is_primary():
    run["train/transfformer-val_loss"].log(val_loss)
    return val_loss

# --------------------------
# Transformer (DistilBERT)
# --------------------------

if args.train_transformer:
    print("Training transformer...")
#    for i in range(args.epoch):
#        train(i, train_dataloader, model, optimizer, scheduler, device)
    fit_distil_with_embeds(
        model=model,
        train_inputs_embeds=x_tr,#.to(device),
        train_labels=y_tr,#.to(device, dtype=torch.long),
        val_inputs_embeds=x_val,#.to(device),
        val_labels=y_val,#.to(device, dtype=torch.long),
        lr=0.001,
        weight_decay=1e-2,
        n_epochs=800,
        batch_size=16,
        n_warmup_epochs=10,
        ckpt_dir="mdl_checkpoints",
        resume_weights="distilBERT_weights.pt",
        device=device,
    )
    if dist.is_primary():
        ckpt_path = f"/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch{str(j).zfill(3)}.pt"
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)
    val_loss = evaluate_transformer(model, val_dataloader, device, run)
else:
    print(f"Loading transformer from {args.ckpt_distil_combined}")
    model.load_state_dict(torch.load(args.ckpt_distil_combined, map_location=device))
    model.eval()



def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Map tensor to [0,1] for display; works for [-1,1] or arbitrary ranges."""
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return x
    xmin = x.amin()
    xmax = x.amax()
    # if already within [-1,1], do a quick mapping
    if xmin >= -1.0 and xmax <= 1.0:
        x = (x + 1.0) / 2.0
    else:
        # per-sample min-max
        dims = tuple(range(1, x.ndim))
        x = (x - x.amin(dim=dims, keepdim=True)) / (
            x.amax(dim=dims, keepdim=True) - x.amin(dim=dims, keepdim=True) + 1e-12
        )
    return x.clamp(0, 1)

def plot_distil_out(distil_out: torch.Tensor, epoch, n: int = 3, title: str = "Distil outputs", save_path: str = None):
    """
    distil_out: [B, C, H, W]
    Shows first n images. Supports C=1 (grayscale) or C=3 (RGB).
    """
    assert distil_out.ndim == 4, f"Expected [B,C,H,W], got {tuple(distil_out.shape)}"
    B, C, H, W = distil_out.shape
    n = min(n, B)

    imgs = _to_01(distil_out[:n])  # [n, C, H, W]

    plt.figure(figsize=(3*n, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.axis("off")
        img = imgs[i]
        if C == 1:
            plt.imshow(img[0].numpy(), cmap="gray")
        else:
            plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"sample {i}_{epoch}")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()




# =============================================================================
# # --------------------------
# # Transformer: optional resume
# # --------------------------
# if args.continue_train_transformer:
#     last = _latest_ckpt_in_distil_dir()
#     if last is None:
#         if is_primary(): print("[Transformer] No checkpoint found; starting fresh.")
#     else:
#         if is_primary(): print(f"[Transformer] Resuming from: {last}")
#         state = torch.load(last, map_location=device)
#         if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
#             model.module.load_state_dict(state)
#         else:
#             model.load_state_dict(state)
#         model.train()
# =============================================================================




# =============================================================================
# python pipeline_vqvae-transformer-classification_gpu01_correct_final-automated.py \
#   --train_transformer \
#   --continue_train_transformer \
#   --resume_transformer_path /local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100_fullTrain.pt \
#   --epoch 400
# =============================================================================

# --------------------------
# Transformer training / resume
# --------------------------
start_epoch = 0
if args.continue_train_transformer:
    ckpt_path = args.ckpt_distil_combined
    if ckpt_path is None:
        if is_primary():
            print("[Transformer] No checkpoint found; starting fresh.")
    else:
        if is_primary():
            print(f"[Transformer] Resuming from: {ckpt_path}")
        last_epoch = _load_transformer_ckpt_into(model, optimizer, scheduler, ckpt_path, device)
        start_epoch = last_epoch  # we'll continue from last_epoch+1 below

# =============================================================================
# # Train for --epoch more epochs
# for ep in range(start_epoch + 1, start_epoch + 1 + args.epoch):
#     train(ep-1, train_dataloader, model, optimizer, scheduler, device)  # if your train() expects 0-based
#     torch.cuda.empty_cache()
#     if 'run' in globals() and run is not None and is_primary():
#         run["train/epoch"].log(ep)
# 
#     # (optional) validation
#     # val_loss = evaluate_transformer(model_distil, val_dataloader, device, run)
# 
#     # save checkpoint each epoch (model-only or full)
#     if is_primary():
#         # Example: save full state so next resume restores LR schedule too
#         state = {
#             "epoch": ep,
#             "model": (model.module.state_dict()
#                       if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
#                       else model.state_dict()),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict() if scheduler is not None else None,
#         }
#         out_ckpt = f"/local/altamabp/checkpoint/distil/distil_transformer_epoch{ep:03d}.pt"
#         torch.save(state, out_ckpt)
#         print(f"[Transformer] Saved: {out_ckpt}")
# 
# =============================================================================
# =============================================================================
# # --------------------------
# # Training loop Transformer
# # --------------------------
# if args.train_transformer:
#     print("Training transformer...")
#     j = 0
#     min_validation_loss = float("inf")
#     for i in range(args.epoch):
#         j += 1
#         print(len(train_dataloader))
#         train(i, train_dataloader, model, optimizer, scheduler, device)
#         torch.cuda.empty_cache()
#         run["train/epoch"].log(j)
# 
#         if is_primary():
#             # Save checkpoint
#             ckpt_path = f"/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch{str(j).zfill(3)}_fullTrain.pt"
#             to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
#             torch.save(to_save.state_dict(), ckpt_path)
# 
#         # ---- Transformer validation (MLM loss) ----
#         t_val = evaluate_transformer(model, val_dataloader, device, run)
#         print(f"[Transformer] Validation loss: {t_val:.6f}")
# 
# else:
#     # Do NOT train transformer unless --train_transformer was provided
#     print("Skipping transformer training (no --train_transformer). Loading checkpoint for eval/inference...")
#     state = torch.load(args.ckpt_distil_combined, map_location=device)
#     if isinstance(model, torch.nn.DataParallel):
#         model.module.load_state_dict(state)
#     else:
#         model.load_state_dict(state)
#     model.eval()
# =============================================================================


# =============================================================================
# # Training loop
# j = 0
# min_validation_loss = np.inf
# for i in range(args.epoch):
#     j += 1
#     print(len(train_dataloader))
#     train(i, train_dataloader, model, optimizer, scheduler, device)
#     torch.cuda.empty_cache()
#     run["train/epoch"].log(j)
#     if is_primary():
#         if isinstance(model, torch.nn.DataParallel):
#             model = model.module
#         torch.save(model.state_dict(), f"/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch{str(j).zfill(3)}_fullTrain.pt")
# 
# 
#     # ---- Transformer validation (MLM loss) ----
#     t_val = evaluate_transformer(model, val_dataloader, device, run)
#     print(f"[Transformer] Final validation loss: {t_val:.6f}")
# 
# =============================================================================


# # Run Masking-Transformer-VQVAE

# In[14]:


# =============================================================================
# run = neptune.init_run(
#     project="UTKFaces",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMmExYTliOC1mYjkyLTQ4M2YtYjFiYS1iZWQ1Y2E0OTJlNTkifQ==",
#     capture_stdout = False,
#     capture_stderr = False,
#     source_files=["pipeline_vqvae-transformer-classification_gpu01_correct_final-automated.py"]
# )
# =============================================================================
    


# In[15]:


#def random_mask(unmasked, indices_unmasked,n_sample, n_token, mask_perc):
#    
#    mask = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[mask_perc, 1 - mask_perc])
#    masked = unmasked.clone()
#    #masked[mask] = 0  # Assuming 0 is the mask token
#    masked[mask.unsqueeze(-1).expand_as(masked)] = 0
#
#    indices_masked = indices_unmasked.clone()
#    indices_masked[~mask[0]] = -100 # Assuming -100 is the mask label token
#   
#    return masked, indices_masked, mask[0][0]

#def random_mask(q, index, n_sample, n_token, perc):
#    masked = q.clone()
#    mask = torch.zeros((1, n_token), dtype=torch.bool, device=q.device)
#    
#    num_mask = int(perc * n_token)
#    chosen = torch.randperm(n_token)[:num_mask]
#    mask[0, chosen] = True
#    
#    # Broadcast mask over embedding dim
#    masked[mask.unsqueeze(-1).expand_as(masked)] = 0
#    
#    index_masked = index.clone()
#    index_masked[mask] = 0  # index has shape [n_token], so mask works directly
#    
#    return masked, index_masked, mask

def random_mask(unmasked, indices_unmasked, n_sample, n_token, mask_perc):
    # Boolean mask [n_token]
    mask = (torch.rand((n_token,), device=unmasked.device) < mask_perc)  # shape [n_token]

    masked = unmasked.clone()
    masked[:, mask, :] = 0  # zero out embeddings for masked tokens

    indices_masked = indices_unmasked.clone()
    indices_masked[mask] = 0#-100  # set masked tokens to -100

    return masked, indices_masked, mask


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=1)

port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
parser.add_argument("--batch_size", type=int, default=1)#64)
parser.add_argument('--ckpt_vqvae', type=str, default="/local/altamabp/checkpoint/vqvae/model_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.pth")
parser.add_argument('--ckpt_distil_combined', type=str, default="/local/altamabp/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100_fullTrain.pt")


# added
parser.add_argument("--train_vqvae", action="store_true", help="Train VQ-VAE instead of loading checkpoint")
parser.add_argument("--train_classifier", action="store_true", help="Train classifier instead of loading checkpoint")
parser.add_argument("--train_transformer", action="store_true", help="Train transformer instead of loading checkpoint")

parser.add_argument("--classifier_ckpt", type=str, default="/local/altamabp/checkpoint/classifier/resnet50/weights_epoch100_fullTrain.pth", help="Path to classifier checkpoint")


# In[ ]:


torch.cuda.set_device(2)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
args.distributed = ddp_ready() and (dist.get_world_size() > 1)

indices = np.load('/local/altamabp/checkpoint/vqvae/indices_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
n, h, w = indices.shape
indices = indices.reshape(n, h * w)

quantizes = np.load('/local/altamabp/checkpoint/vqvae/quantized_epoch100_flat_vqvae80x80_144x456codebook_fullTrain.npy')
quant_b = quantizes
n, c, h, w = quantizes.shape
quantizes = quantizes.transpose(0, 2, 3, 1)
quantizes = quantizes.reshape(n, h * w, c)

#Bottom data and parameters
n_sample = quantizes.shape[0]
d_embed_vec = quantizes.shape[2]
n_token = np.prod(quantizes.shape[1])
quantizes = quantizes.reshape((n_sample, n_token, d_embed_vec))
length = int(math.sqrt(n_token))
indices = indices.reshape((n_sample, n_token))
indices_to_sort = set(indices.flatten())
indices_to_sort = sorted(indices_to_sort)
vocab_size = indices_to_sort[-1] + 1

#Define Distilbert model
cfg = DistilBertConfig(
        vocab_size=vocab_size,
        hidden_size=d_embed_vec,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=4,
        max_position_embeddings=n_token
)
model_distil = DistilBertForMaskedLM(cfg).to(device)
model_distil.load_state_dict(torch.load(args.ckpt_distil_combined))
model_distil = model_distil.to(device)
model_distil.eval()

#Define VQVAE model
model_vqvae = FlatVQVAE().to(device)
model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
model_vqvae = model_vqvae.to(device)
model_vqvae.eval()

# Define classifier and load saved model(weights)
weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
classifier = resnet50(pretrained=False)

#redefine the classifier head to 10 outputs (UTKFace classes) instead of 1000 (ImageNet classes)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)

classifier.load_state_dict(torch.load('/local/altamabp/checkpoint/classifier/resnet50/weights_epoch100_fullTrain.pth'))#30.pth'))
classifier.to(device)
classifier.eval()

mask_percentages = np.arange(0.1, 1.1, 0.1)
# mask_percentages = np.append(mask_percentages,[.85,.95])
mask_percentages = np.sort(mask_percentages)

average_errors = []


import os

# Make sure save directory exists
save_dir = "image/image-reconstructed_fullTrain_800"
os.makedirs(save_dir, exist_ok=True)

criterion = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()



# Rebuild the images list in the same order used when encoding:
# (this mirrors how 'batch_indices' was formed originally)
all_paths = data_loader.dataset.image_files              # full list in order
img_names = [os.path.basename(p) for p in all_paths]     # or keep full paths

# Now use img_names[x] exactly as above



#for x in range(quantizes.shape[0]):
for x in tqdm(range(quantizes.shape[0]), desc="Processing", unit="img"):
    #print(x)

    transformer_outputs = []

    q = torch.from_numpy(quantizes[x]).to(device)
    index = torch.from_numpy(indices[x]).to(device)
    q = q.reshape(1, q.size(0), q.size(1))

    # Decode original image
    vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device))

    # Decode masked once at 50% for visualization
    with torch.no_grad():
        q_masked, index_masked, mask = random_mask(q, index, n_sample, n_token, 0.5)
        index_masked_forvis = index.clone()
        index_masked_forvis[mask] = 0
        vqvae_masked_out = model_vqvae.decode_code(index_masked_forvis.reshape(1, length, length))


    # Classify original for naming
    vqvae_img = (vqvae_out.clamp(-1, 1) + 1) / 2.0 # -> [0,1] #preprocess(vqvae_out.unsqueeze(0)).to(device)
    vqvae_img_prob = classifier(vqvae_img) #classifier(preprocess(vqvae_img).unsqueeze(0).to(device))
    _, vqvae_img_label = torch.max(vqvae_img_prob, 1)


    # Loop over mask percentages
    for perc in mask_percentages:
        with torch.no_grad():
            q_masked, index_masked, mask = random_mask(q, index, n_sample, n_token, perc)
            q_masked = q_masked.to(device)
        
            outputs = model_distil(inputs_embeds=q_masked, output_hidden_states=True)
        
            # 1) argmax -> tokens
            pred_tokens = torch.argmax(outputs.logits, dim=2)[0].to(device).long()
        
            # 2) clamp to valid vocab (defensive)
            vmax = model_distil.config.vocab_size - 1
            pred_tokens.clamp_(0, vmax)
        
            # 3) copy original index grid, fill only masked positions, ensure long dtype
            confidence_based_recons_index = index.clone().long()
            confidence_based_recons_index[mask] = pred_tokens[mask]
            
            # Confirm shapes and dtypes before decoding
            assert confidence_based_recons_index.dtype == torch.long
            assert confidence_based_recons_index.shape == (n_token,)
        
            # 4) decode using LONG index grid of shape [1, H, W]
            distil_out = model_vqvae.decode_code(
                confidence_based_recons_index.reshape(1, length, length).long()
            )
        
            transformer_outputs.append(distil_out.squeeze(0).cpu())

# =============================================================================
#         with torch.no_grad():
#             q_masked, index_masked, mask = random_mask(q, index, n_sample, n_token, perc)
#             q_masked = q_masked.to(device)
# 
#             outputs = model_distil(inputs_embeds=q_masked, output_hidden_states=True)
#             pred_tokens = torch.argmax(outputs.logits, dim=2)[0].detach().to(device).long()
# 
#             confidence_based_recons_index = index.clone()
#             confidence_based_recons_index[mask] = pred_tokens[mask]
# 
#             distil_out = model_vqvae.decode_code(
#                 confidence_based_recons_index.reshape(1, length, length)
#             )
# 
#             transformer_outputs.append(distil_out.squeeze(0).cpu())
# 
# =============================================================================
            
            # Print the fraction of tokens predicted as 0. If too high, transformer needs more training /  lower mask rate.
            if perc in (0.5, 0.9):
                zero_frac = (pred_tokens == 0).float().mean().item()
                if zero_frac > 0.5 and is_primary():
                    print(f"[warn] perc={perc:.2f}: {zero_frac*100:.1f}% tokens predicted as 0")
    
    # If you saved full paths, this will just return immediately.
    # If you saved basenames, this will resolve them under your dataset roots.
    # --- resolve path (works for full paths or basenames) ---
    img_path = resolve_image_path(
        img_names[x],
        roots=[
            "/local/altamabp/UTKFace_dataset_subset_150000",
            "/local/altamabp/UTKFace_dataset_test_structured",
        ],
    )
    
    # --- load the image ---
    img = Image.open(img_path).convert("RGB")
    
    # --- target HW = VQ-VAE recon size (e.g., 80x80) ---
    target_hw = tuple(vqvae_out.shape[-2:])  # (H, W)
    
    # --- prefer dataset transform if present AND produces correct size; else fallback ---
    vq_tfm = getattr(train_dataset_vq, "transform", None) or getattr(val_dataset_vq, "transform", None)
    if vq_tfm is not None:
        probe = vq_tfm(img)
        if tuple(probe.shape[-2:]) == target_hw:
            orig = probe
        else:
            # dataset tf outputs wrong size (e.g., token grid 20x20) -> fallback to a single, correct resize
            img_rs = TF.resize(img, target_hw, interpolation=InterpolationMode.BICUBIC, antialias=True)
            orig = TF.to_tensor(img_rs)
            orig = TF.normalize(orig, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
    else:
        # no dataset tf -> explicit resize + normalize once
        img_rs = TF.resize(img, target_hw, interpolation=InterpolationMode.BICUBIC, antialias=True)
        orig = TF.to_tensor(img_rs)
        orig = TF.normalize(orig, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
    

    # --- build row: ORIGINAL | VQ-VAE RECON | MASKED | transformer recons ---
    row_images = [orig.cpu(),
                  vqvae_out.squeeze(0).cpu(),
                  vqvae_masked_out.squeeze(0).cpu()] + transformer_outputs

    # Labels: no text for first two, show mask % for masked + transformer outputs
    masked_pct = 0.5  # this is the one you used above for vqvae_masked_out
    labels = ["Original", "VQVAE recon"] + [f"mask {int(masked_pct*100)}%"] + [f"mask {int(p*100)}%" for p in mask_percentages]
    
    # Save only every 10,000th sample
    if x % 10000 == 0:
        print('max vqvae out: ', vqvae_out.max().item())
        print('min vqvae_out: ', vqvae_out.min().item())
        print('max orig: ', orig.max().item())
        print('min orig: ', orig.min().item())
        
        out_path = os.path.join(
            save_dir,
            f"80x80_random_{str(x).zfill(5)}_{vqvae_img_label.item()}.png"
        )
        save_row_with_labels(row_images, labels, out_path)  # <-- new saver with text
        print(f"Saved: {out_path}")





# for plotting the three panels (original, masked, transformer) for one masking percentag --> for plotting more masking percentages per plot use the previous code block
# =============================================================================
# for perc in mask_percentages:
#     reconstruction_errors = []
#     cross_entropy_class_err = []
#     correct_random_pred = 0
#     tot_sample = 0
# 
#     criterion = nn.MSELoss()
#     criterion_class = nn.CrossEntropyLoss()
# 
#     for x in range(quantizes.shape[0]):
#         print(x)
#         q = torch.from_numpy(quantizes[x]).to(device)
#         index = torch.from_numpy(indices[x]).to(device)
#         q = q.reshape(1, q.size(0), q.size(1))
# 
#         with torch.no_grad():
#             q_masked, index_masked, mask = random_mask(q, index, n_sample, n_token, perc)
#             q_masked = q_masked.to(device)
#             index_masked = index_masked.to(device)
# 
#         with torch.no_grad():
#             outputs = model_distil(inputs_embeds=q_masked, output_hidden_states=True)
#             confidence_based_prediction = torch.argmax(outputs.logits, dim=2)  # [1, n_token]
# 
#             # Clone and replace masked tokens with predictions
#             confidence_based_recons_index = index.clone()
#             pred_tokens = confidence_based_prediction[0].detach().to(device).long()
# 
#             # Debug check
#             #print(f"[DEBUG] Sample {x}, perc={perc:.2f}, pred min={pred_tokens.min().item()}, pred max={pred_tokens.max().item()}, vocab={model_distil.config.vocab_size}")
# 
#             confidence_based_recons_index[mask] = pred_tokens[mask]
# 
#             # Decode transformer reconstruction
#             distil_out = model_vqvae.decode_code(
#                 confidence_based_recons_index.reshape(1, length, length)
#             )
# 
#             # Decode original reconstruction
#             vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device))
# 
#             # Decode masked reconstruction for visualization
#             index_masked_forvis = index.clone()
#             index_masked_forvis[mask] = 0
#             vqvae_masked_out = model_vqvae.decode_code(
#                 index_masked_forvis.reshape(1, length, length)
#             )
# 
#             # Classification
#             vqvae_img = preprocess(vqvae_out.unsqueeze(0)).to(device)
#             vqvae_img_prob = classifier(vqvae_img)
#             _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
# 
#             rand_mask_img = preprocess(distil_out).to(device)
#             rand_mask_img_prob = classifier(rand_mask_img)
#             _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
# 
#             correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
#             tot_sample += 1
# 
#             
#             # Always save first sample of each percentage and then every 500th after
#             if x % 500 == 0 or x == 0:
#                 out_path = os.path.join(
#                     save_dir,
#                     f"80x80_random_{str(x).zfill(5)}_{int(perc*100)}_{vqvae_img_label.item()}_{rand_mask_img_label.item()}.png"
#                 )
#                 
#                 # Ensure all outputs are [C, H, W]
#                 vqvae_out = vqvae_out.squeeze(0)
#                 vqvae_masked_out = vqvae_masked_out.squeeze(0)
#                 distil_out = distil_out.squeeze(0)
#                 
#                 # Stack into (N, C, H, W)
#                 img = torch.stack([vqvae_out, vqvae_masked_out, distil_out], dim=0).cpu()
#                 
#                 #img = torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).cpu()
#                 utils.save_image(img, out_path, nrow=3, normalize=True)
#                 print(f"Saved: {out_path}")
# 
#         # Errors
#         recon_loss = criterion(distil_out, vqvae_out)
#         run["recons/mse_per_image_random_mask"].log(recon_loss.item())
#         reconstruction_errors.append(recon_loss.item())
# 
#         class_loss = criterion_class(rand_mask_img_prob, vqvae_img_prob)
#         run["recons/cross_entropy_per_image_random_mask"].log(class_loss.item())
#         cross_entropy_class_err.append(class_loss.item())
# 
#     # Log averages
#     run["recons/average_mse_per_precision_random_mask"].log(np.mean(reconstruction_errors))
#     run["recons/average_cross_entropy_error_random_mask"].log(np.mean(cross_entropy_class_err))
#     pred_acc_random_mask = correct_random_pred / tot_sample
#     run["recons/average_classification_accuracy_random"].log(1 - pred_acc_random_mask)
# =============================================================================


# =============================================================================
# for perc in mask_percentages:
#     reconstruction_errors = []
#     cross_entropy_class_err = []
# 
#     criterion = nn.MSELoss()
#     criterion_class = nn.CrossEntropyLoss()
#     correct_random_pred = 0
#     tot_sample = 0
#     for x in range(0,quantizes.shape[0]):
#         print(x)
#         q = torch.from_numpy(quantizes[x])
#         index = torch.from_numpy(indices[x])
#         index = index.to(device)
#         q = q.to(device)
#         q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
#         
#         with torch.no_grad():
#             q_masked, index_masked, mask = random_mask(q, index , n_sample, n_token,perc)                                
#             q_masked = q_masked.to(device)
#             index_masked = index_masked.to(device)
# 
#         #Fill in predicted tokens
#         with torch.no_grad():
#             outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
#             confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
#             confidence_based_recons_index =  index.clone()
#             print(mask.shape)
#             for p in range(0,n_token):
#                 if(mask[p]):
#                     #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
#                     #confidence_based_recons_index[p] = confidence_based_prediction[0][p]
#                     confidence_based_recons_index[p] = confidence_based_prediction[0, p] 
#             
#             #Reconstruct with distil predictions
#             confidence_based_recons_index = confidence_based_recons_index.to(device)
#             #distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))
#             distil_out = model_vqvae.decode_code(confidence_based_recons_index.reshape(1, length, length))
# 
#             #Reconstruct Original
#             vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
#             index_masked_forvis = index.clone()
#             index_masked_forvis[mask] = 0
#             #vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
#             vqvae_masked_out = model_vqvae.decode_code(index_masked_forvis.reshape(1, length, length))
# 
#             # Label outputs
#             vqvae_out = vqvae_out.unsqueeze(0)
#             vqvae_img = preprocess(vqvae_out)
#             vqvae_img = vqvae_img.to(device)
#             vqvae_img_prob = classifier(vqvae_img)
#             _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
#             
#             rand_mask_img = preprocess(distil_out)
#             rand_mask_img = rand_mask_img.to(device)
#             rand_mask_img_prob = classifier(rand_mask_img)
#             _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
#             correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
#             print(f'rand_mask_img_label is {rand_mask_img_label}')
#             print(f'vqvae_img_label is {vqvae_img_label}')
#             print(f'rand_mask_img_label is {rand_mask_img_label.item()}')
#             print(f'vqvae_img_label is {vqvae_img_label.item()}')
#             print(f'correct_random_pred is {correct_random_pred}')
#             tot_sample += 1
#             print(tot_sample)
# 
# 
# #            if x%5 ==0:
#             utils.save_image(
#                 torch.cat([vqvae_out, vqvae_masked_out, distil_out], 0).to(device),
#                 f"image/mask-reconstructed/80x80_random_{str(x).zfill(5)}_{int(perc*100)}_{vqvae_img_label.item()}_{rand_mask_img_label.item()}.png",
#                 nrow=3,
#                 normalize=True,
#                 #range=(-1, 1),
#             )
#         
#         recon_loss = criterion(distil_out, vqvae_out)
#         run["recons/mse_per_image_random_mask"].log(recon_loss.item())
#         reconstruction_errors.append(recon_loss.item())
#         class_loss = criterion_class(rand_mask_img_prob, vqvae_img_prob)
#         run["recons/cross_entropy_per_image_random_mask"].log(class_loss.item())
#         cross_entropy_class_err.append(class_loss.item())
#     run["recons/average_mse_per_precision_random_mask"].log(np.mean(reconstruction_errors))
#     run["recons/average_cross_entropy_error_random_mask"].log(np.mean(cross_entropy_class_err))
#     average_errors.append(np.mean(reconstruction_errors))
#     pred_acc_random_mask = correct_random_pred/tot_sample
#     pred_err_random_mask = 1-pred_acc_random_mask
#     run["recons/average_classification_accuracy_random"].log(pred_err_random_mask)
# 
# 
# =============================================================================
