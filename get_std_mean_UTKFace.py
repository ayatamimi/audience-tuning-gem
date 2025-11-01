# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 18:48:03 2025

@author: ayata
"""

#!/usr/bin/env python3
import math, argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import glob, os

# Use only basic prep: Resize (optional), ToTensor (puts pixels in [0,1])
def make_transform(size=None):
    ops = []
    if size is not None:
        ops += [transforms.Resize((size, size))]
    ops += [transforms.ToTensor()]  # HWC [0..255] -> CHW float [0..1]
    return transforms.Compose(ops)

# Flat dataset (no class subfolders) 
class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None, extensions=("jpg","jpeg","png","bmp","webp")):
        self.root = str(root)
        self.transform = transform
        pats = [os.path.join(self.root, f"**/*.{ext}") for ext in extensions]
        files = []
        for p in pats:
            files.extend(glob.glob(p, recursive=True))
        self.files = [f for f in files if os.path.isfile(f)]
        if not self.files:
            raise FileNotFoundError(f"No images found under: {self.root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return (self.transform(img) if self.transform else img), 0

def compute_mean_std(dataset, batch_size=256, num_workers=8):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    n_pixels = 0
    mean = torch.zeros(3)
    mean_sq = torch.zeros(3)

    for x, _ in tqdm(loader, desc="Pass 1/1: accumulating"):
        # x: [B, C, H, W] with values in [0,1]
        b, c, h, w = x.shape
        pixels = b * h * w
        n_pixels += pixels

        # sum over batch+spatial dims
        sum_ = x.sum(dim=[0, 2, 3])           # [C]
        sum_sq = (x ** 2).sum(dim=[0, 2, 3])  # [C]
        mean += sum_
        mean_sq += sum_sq

    mean /= n_pixels
    var = (mean_sq / n_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="UTKFace root (classed or flat)")
    ap.add_argument("--flat", action="store_true", help="Use FlatImageDataset (no class subfolders)")
    ap.add_argument("--size", type=int, default=80, help="Optional resize (H=W=size)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tfm = make_transform(size=args.size)

    if args.flat:
        ds = FlatImageDataset(args.root, transform=tfm)
    else:
        # For classed folders (e.g., 0..9 subfolders)
        ds = datasets.ImageFolder(args.root, transform=tfm)

    mean, std = compute_mean_std(ds, batch_size=args.batch, num_workers=args.workers)
    print(f"mean = {mean}")
    print(f"std  = {std}")
    print("\nUse as:")
    print(f"transforms.Normalize(mean={mean}, std={std})")

# calculated std and mean of UTKFace
#transforms.Normalize(mean=[0.6154290437698364, 0.46279090642929077, 0.38601234555244446], std=[0.24672381579875946, 0.22112978994846344, 0.21502047777175903])
