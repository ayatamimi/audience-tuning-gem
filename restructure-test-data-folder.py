# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 23:56:16 2025

@author: ayata
"""
import os
import re
import shutil

# Paths
src_dir = "UTKFace_dataset_subset_150000"
dst_dir = "UTKFace_dataset_subset_150000_structured"

# Create destination root
os.makedirs(dst_dir, exist_ok=True)

# Regex to extract last digit (class label)
pattern = re.compile(r"^(\d+)_.*_(\d)\.png$")

for fname in os.listdir(src_dir):
    if not fname.endswith(".png"):
        continue

    match = pattern.match(fname)
    if not match:
        print(f"Skipping {fname} (filename pattern not matched)")
        continue

    label = match.group(2)  # last digit = class label (0–9)

    # Create class subfolder if not exists
    class_dir = os.path.join(dst_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    # Move (or copy) file
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(class_dir, fname)
    shutil.copy(src_path, dst_path)   # use .move() if you want to relocate

print("✅ Restructuring complete. Images now stored in", dst_dir)
