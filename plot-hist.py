# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:35:43 2025

@author: ayata
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# load indices (true) directly
indices = np.load("./checkpoint_correct/vqvae//train_codebook_vqvae_80x80_codebook_16x124.npy")
true = np.asarray(indices, dtype=np.int64).ravel()

# build bins 0..max(true)
num_bins = int(true.max() + 1) if true.size else 1
idx  = np.arange(num_bins)
bins = np.arange(num_bins + 1) - 0.5  # center bars on integer indices

true_counts, _ = np.histogram(true, bins=bins)

# plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(idx, true_counts, width=0.8, label=f"True (total={true_counts.sum()})")
ax.set_xlabel("Codebook Index")
ax.set_ylabel("Count per Index")
ax.set_title("Index Distribution (True Only)")
ax.legend()
fig.tight_layout()

# save
save_path = Path("./image_correct/vqvae_reconstruction_16x24/codebook_hist_124x16vqvae.png")
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)
