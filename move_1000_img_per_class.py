# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:48:28 2025

@author: ayata
"""

# =============================================================================
# # Move exactly up to 1000 PNGs per class, ensuring they’re removed from source
# python move_per_class_strict_move.py /path/to/src_root /path/to/dest_root
# 
# # Random selection, reproducible, and fail if any leftover file can't be deleted
# python move_per_class_strict_move.py /path/to/src_root /path/to/dest_root --random --seed 42 --strict
# 
# # Preview without changes
# python move_per_class_strict_move.py /path/to/src_root /path/to/dest_root --dry-run
# 
# =============================================================================


#!/usr/bin/env python3
import argparse
import random
import shutil
import os
from pathlib import Path

CLASSES = [str(i) for i in range(10)]

def list_pngs(folder: Path):
    # Non-recursive, case-insensitive .png
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]

def unique_destination(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        cand = dest_dir / f"{stem}__moved{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def safe_move_and_ensure_removed(src: Path, dst: Path, strict: bool) -> None:
    """
    Move src -> dst.
    1) Try Path.rename (atomic if same filesystem).
    2) If that fails, fall back to shutil.move (copy+delete across fs).
    3) Verify src no longer exists; if it does, try to unlink it.
       - If still present and strict=True, raise an error.
    """
    # Make sure destination parent exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        src.rename(dst)  # fastest if same filesystem
    except Exception:
        # Cross-device or permission issue; fall back
        shutil.move(str(src), str(dst))

    # Verify removal from source
    if src.exists():
        try:
            src.unlink()  # force delete leftover
        except Exception as e:
            if strict:
                raise RuntimeError(f"After moving, source still exists and could not be deleted: {src} ({e})")
            else:
                print(f"[WARN] Source still existed after move; attempted delete but failed: {src} ({e})")

def main():
    ap = argparse.ArgumentParser(
        description="Move N PNGs per class (0-9) from source to destination, ensuring removal from source."
    )
    ap.add_argument("src", type=Path, help="Source root containing subfolders 0..9")
    ap.add_argument("dest", type=Path, help="Destination root; subfolders 0..9 will be created")
    ap.add_argument("-n", "--count", type=int, default=1000, help="Images to move per class (default: 1000)")
    ap.add_argument("--random", action="store_true", help="Randomly choose images instead of name order")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (used with --random)")
    ap.add_argument("--dry-run", action="store_true", help="List planned moves without changing files")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any selected file cannot be fully removed from source after moving")
    args = ap.parse_args()

    src_root: Path = args.src
    dest_root: Path = args.dest
    per_class = args.count

    if not src_root.is_dir():
        raise SystemExit(f"Source is not a directory: {src_root}")

    # Validate classes in source and prepare destination
    missing = [c for c in CLASSES if not (src_root / c).is_dir()]
    if missing:
        raise SystemExit(f"Missing class subfolders in source: {', '.join(missing)}")

    for c in CLASSES:
        (dest_root / c).mkdir(parents=True, exist_ok=True)

    if args.random and args.seed is not None:
        random.seed(args.seed)

    grand_total = 0
    for c in CLASSES:
        sdir = src_root / c
        ddir = dest_root / c

        files = list_pngs(sdir)
        if not files:
            print(f"[{c}] No PNGs found. Skipping.")
            continue

        # Choose up to per_class
        chosen = (random.sample(files, k=min(per_class, len(files)))
                  if args.random else
                  sorted(files, key=lambda p: p.name)[:per_class])

        print(f"[{c}] Found {len(files)} PNGs; moving {len(chosen)} -> {ddir}")

        if args.dry_run:
            for p in chosen[:5]:
                print(f"  [DRY-RUN] {p.name} -> {ddir / p.name}")
            if len(chosen) > 5:
                print(f"  [DRY-RUN] ... and {len(chosen) - 5} more")
            grand_total += len(chosen)
            continue

        moved_ok = 0
        for i, p in enumerate(chosen, 1):
            target = unique_destination(ddir, p.name)
            safe_move_and_ensure_removed(p, target, strict=args.strict)
            moved_ok += 1
            if i % 500 == 0 or i == len(chosen):
                print(f"  Moved {i}/{len(chosen)}")

        # Optional: sanity check of counts
        after_files = list_pngs(sdir)
        delta = len(files) - len(after_files)
        print(f"[{c}] Done. Confirmed reduction by {delta}.")
        grand_total += moved_ok

    print(f"All classes complete. Total moved: {grand_total}")

if __name__ == "__main__":
    main()



#==============================================================================
#move without deletion from source
# =============================================================================
# # Deterministic by filename (moves up to 1000 per class)
# python move_per_class.py /path/to/src_root /path/to/dest_root
# 
# # Random selection, reproducible
# python move_per_class.py /path/to/src_root /path/to/dest_root --random --seed 42
# 
# # Preview without moving files
# python move_per_class.py /path/to/src_root /path/to/dest_root --dry-run
# =============================================================================

# =============================================================================
# #!/usr/bin/env python3
# import argparse
# import random
# import shutil
# from pathlib import Path
# 
# CLASSES = [str(i) for i in range(10)]
# 
# def list_pngs(folder: Path):
#     # non-recursive, case-insensitive .png
#     return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
# 
# def unique_destination(dest_dir: Path, name: str) -> Path:
#     target = dest_dir / name
#     if not target.exists():
#         return target
#     stem, suffix = target.stem, target.suffix
#     i = 1
#     while True:
#         cand = dest_dir / f"{stem}__moved{i}{suffix}"
#         if not cand.exists():
#             return cand
#         i += 1
# 
# def main():
#     ap = argparse.ArgumentParser(
#         description="Move N PNG images per class (0-9) from source to destination, preserving class folders."
#     )
#     ap.add_argument("src", type=Path, help="Source root that contains subfolders 0..9")
#     ap.add_argument("dest", type=Path, help="Destination root to create subfolders 0..9")
#     ap.add_argument("-n", "--count", type=int, default=1000, help="Images to move per class (default: 1000)")
#     ap.add_argument("--random", action="store_true", help="Randomly choose images instead of name order")
#     ap.add_argument("--seed", type=int, default=None, help="Random seed (used with --random)")
#     ap.add_argument("--dry-run", action="store_true", help="Show planned moves without changing files")
#     args = ap.parse_args()
# 
#     src_root: Path = args.src
#     dest_root: Path = args.dest
#     per_class = args.count
# 
#     # Validate source
#     if not src_root.is_dir():
#         raise SystemExit(f"Source is not a directory: {src_root}")
# 
#     # Ensure class subfolders exist in src and create in dest
#     missing = []
#     for c in CLASSES:
#         if not (src_root / c).is_dir():
#             missing.append(c)
#     if missing:
#         raise SystemExit(f"Missing class subfolders in source: {', '.join(missing)}")
# 
#     for c in CLASSES:
#         (dest_root / c).mkdir(parents=True, exist_ok=True)
# 
#     if args.random and args.seed is not None:
#         random.seed(args.seed)
# 
#     grand_total = 0
#     for c in CLASSES:
#         sdir = src_root / c
#         ddir = dest_root / c
# 
#         files = list_pngs(sdir)
#         if not files:
#             print(f"[{c}] No PNGs found. Skipping.")
#             continue
# 
#         if args.random:
#             chosen = random.sample(files, k=min(per_class, len(files)))
#         else:
#             chosen = sorted(files, key=lambda p: p.name)[:per_class]
# 
#         print(f"[{c}] Found {len(files)} PNGs; moving {len(chosen)} to {ddir}")
# 
#         if args.dry_run:
#             for p in chosen[:5]:
#                 print(f"  [DRY-RUN] {p.name} -> {ddir / p.name}")
#             if len(chosen) > 5:
#                 print(f"  [DRY-RUN] ... and {len(chosen) - 5} more")
#             grand_total += len(chosen)
#             continue
# 
#         moved = 0
#         for i, p in enumerate(chosen, 1):
#             target = unique_destination(ddir, p.name)
#             shutil.move(str(p), str(target))
#             moved += 1
#             if i % 500 == 0 or i == len(chosen):
#                 print(f"  Moved {i}/{len(chosen)}")
# 
#         print(f"[{c}] Done. Moved {moved}.")
#         grand_total += moved
# 
#     print(f"All classes complete. Total moved: {grand_total}")
# 
# if __name__ == "__main__":
#     main()
# 
# =============================================================================
