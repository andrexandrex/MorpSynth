import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,ConcatDataset
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_trans
from torchvision.transforms import InterpolationMode
import cv2
# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio

mean = [0.4854, 0.4854, 0.4854]
std  = [0.1782, 0.1782, 0.1782]

augmented_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])

plain_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])

valid_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])



class M4DSAROilSpillDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # List all .tif patch files
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith('.tif') and '_patch_' in f
        ])

        # Derive corresponding label filenames
        self.label_files = [
            f.replace('_patch_', '_mask1d_').replace('.tif', '.png')
            for f in self.image_files
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_fn = self.image_files[idx]
        lbl_fn = self.label_files[idx]

        img_path = os.path.join(self.images_dir, img_fn)
        lbl_path = os.path.join(self.labels_dir, lbl_fn)

        # Read .tif image using rasterio (SAR-specific)
        with rasterio.open(img_path) as src:
            image = src.read()  # Shape: (C, H, W)
            image = np.nan_to_num(image)  # Handle NaNs if present

        # Normalize or scale here if needed

        # Convert to float32 torch tensor
        #image = torch.as_tensor(image, dtype=torch.float32)

        # Read label (grayscale mask)
        label = np.array(Image.open(lbl_path).convert('L'))
        #label = torch.as_tensor(label, dtype=torch.long)

        # Apply transform if any (Albumentations-style)
        if self.transform:
            # Convert image to (H, W, C) for transform
            image_hwc = np.transpose(image, (1, 2, 0)).astype(np.float32)
            augmented = self.transform(image=image_hwc, mask=label)
            # ToTensorV2 already returns a C×H×W torch.Tensor
            image = augmented['image']
            label = augmented['mask'].long()

        return image, label

def get_dataloaders_for_training(
    krest_dir,
    batch_size,
    num_workers,
    random_state=None,
    hardneg_csv=None,          # NEW: optional CSV path
    frac_hard=0.30,            # NEW: target hard-neg fraction in each epoch
    flat_weight=3.0            # NEW: upweight factor for hard-negatives
):
    train_img_dir  = os.path.join(krest_dir, 'train', 'images')
    train_lbl_dir  = os.path.join(krest_dir, 'train', 'labels_1D')

    aug_ds   = M4DSAROilSpillDataset(train_img_dir, train_lbl_dir, transform=augmented_tf)
    valid_ds = M4DSAROilSpillDataset(train_img_dir, train_lbl_dir, transform=valid_tf)

    # ---- Split (5% val) ----
    n = len(aug_ds)
    idx = np.arange(n)
    split = int(0.05 * n)
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    else:
        np.random.shuffle(idx)
    train_idx, val_idx = idx[split:], idx[:split]

    # ---- Build validation loader (simple subset) ----
    valid_subset = Subset(valid_ds, val_idx)
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # ---- Build training subset ----
    train_subset = Subset(aug_ds, train_idx)

    # ---- Default: uniform sampling ----
    weights = np.ones(len(train_subset), dtype=np.float32)

    # ---- If we have a hard-negative list, upweight them ----
    if hardneg_csv is not None and os.path.exists(hardneg_csv):
        df = pd.read_csv(hardneg_csv)
        if 'image_file' not in df.columns:
            raise ValueError("hardneg_csv must contain a column named 'image_file'")

        # Map filename -> position inside the *train subset*
        name_to_pos = {}
        for pos, ds_i in enumerate(train_idx):
            name_to_pos[aug_ds.image_files[ds_i]] = pos

        # Positions of hard-negatives that are actually present in the train subset
        hard_pos = []
        for fn in df['image_file'].astype(str).tolist():
            pos = name_to_pos.get(fn)
            if pos is not None:
                hard_pos.append(pos)

        hard_pos = np.array(hard_pos, dtype=int)

        if hard_pos.size > 0:
            weights = np.ones(len(train_subset), dtype=np.float32)
            n_hard_found = int(hard_pos.size)
            n_hard = hard_pos.size
            n_non  = len(train_subset) - n_hard

            if n_non > 0:
                # Choose multiplier k so that:
                #   (k * flat_weight * n_hard) / ((k * flat_weight * n_hard) + (1 * n_non)) ≈ frac_hard
                # => k = (frac_hard / (1 - frac_hard)) * (n_non / (flat_weight * n_hard))
                k = (frac_hard / max(1.0 - frac_hard, 1e-8)) * (n_non / max(flat_weight * n_hard, 1e-8))
            else:
                k = 1.0  # degenerate case: everything is hard

            weights[hard_pos] = float(flat_weight) * float(k)

            # Optional: sanity print for expected fraction (not exact due to replacement & batching)
            num = (weights[hard_pos].sum())
            den = (weights.sum())
            exp_frac = float(num / (den + 1e-12))
            print(f"[sampler] hard_neg target={frac_hard:.3f}, expected≈{exp_frac:.3f}, "
                f"n_hard={n_hard}, n_non={n_non}")
        else:
            print("[sampler] No hard-negatives found in the train subset for the provided CSV.")

    # ---- Sampler draws len(train_subset) items with replacement each epoch ----
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(train_subset), replacement=True
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,               # <— controls composition
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    print(f"Dataset: train {len(train_idx)}, valid {len(val_idx)}")
    if hardneg_csv is not None and os.path.exists(hardneg_csv):
        print(f"Hard-negatives present in train subset: {n_hard_found} / {len(train_subset)} "
            f"(target frac ≈ {frac_hard:.2f})")
    return train_loader, valid_loader