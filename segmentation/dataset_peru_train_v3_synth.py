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
            # ToTensorV2 already returns a CÃ—HÃ—W torch.Tensor
            image = augmented['image']
            label = augmented['mask'].long()

        return image, label
# NEW: Synthetic (PNG) dataset with mask suffix normalization
class M4DSAROilSpillSynthDataset(Dataset):
    """
    Assumes images/ and labels1d/ contain PNGs with identical filenames.
    Tag must be either 'lookalike' or 'oil' (enforced).
    """
    def __init__(self, images_dir, labels_dir, transform=None, strict=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform  = transform
        self.strict     = strict

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])

        def infer_tag(fn: str) -> str:
            s = fn.lower()
            if 'lookalike' in s:
                return 'lookalike'
            if 'oil' in s:
                return 'oil'
            raise ValueError(f"[synth] filename must contain 'lookalike' or 'oil': {fn}")

        self.label_files = []
        self.tags = []
        for f in self.image_files:
            lbl = os.path.join(labels_dir, f)
            if self.strict and not os.path.exists(lbl):
                raise FileNotFoundError(f"[synth] Label not found for image '{f}': {lbl}")
            self.label_files.append(f)
            self.tags.append(infer_tag(f))

        assert len(self.image_files) == len(self.label_files) == len(self.tags)

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_fn = self.image_files[idx]; lbl_fn = self.label_files[idx]
        img_path = os.path.join(self.images_dir, img_fn)
        lbl_path = os.path.join(self.labels_dir,  lbl_fn)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None: raise FileNotFoundError(img_path)
        if image.ndim == 2: image = image[..., None]
        if image.shape[2] == 1: image = np.repeat(image, 3, axis=2)
        image = image.astype(np.float32)

        mask = np.array(Image.open(lbl_path).convert('L'))

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out['image'], out['mask'].long()
        return image, mask, self.tags[idx]


def get_dataloaders_for_training(
    krest_dir,               # keep name as you requested; this is PERU root
    batch_size,
    num_workers,
    random_state=None,
    hardneg_csv=None,        # real hard-negatives list (CSV)
    frac_hard=0.30,
    flat_weight=3.0,
    # NEW: synthetic roots
    synth_root=None,         # e.g., os.path.join(krest_dir, 'train_synth')
    synth_frac_targets=None  # dict of desired fractions by tag
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

    # ---- Validation loader ----
    valid_subset = Subset(valid_ds, val_idx)
    valid_loader = DataLoader(
        valid_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # ---- REAL training subset + sampler ----
    train_subset = Subset(aug_ds, train_idx)
    weights_real = np.ones(len(train_subset), dtype=np.float32)

    n_hard_found = 0
    if hardneg_csv is not None and os.path.exists(hardneg_csv):
        df = pd.read_csv(hardneg_csv)
        if 'image_file' not in df.columns:
            raise ValueError("hardneg_csv must contain a column named 'image_file'")

        name_to_pos = {aug_ds.image_files[ds_i]: pos for pos, ds_i in enumerate(train_idx)}
        hard_pos = [name_to_pos[fn] for fn in df['image_file'].astype(str).tolist()
                    if fn in name_to_pos]
        hard_pos = np.array(hard_pos, dtype=int)
        if hard_pos.size > 0:
            n_hard_found = int(hard_pos.size)
            n_hard = hard_pos.size
            n_non  = len(train_subset) - n_hard
            if n_non > 0:
                k = (frac_hard / max(1.0 - frac_hard, 1e-8)) * (n_non / max(flat_weight * n_hard, 1e-8))
            else:
                k = 1.0
            weights_real[hard_pos] = float(flat_weight) * float(k)

            num = (weights_real[hard_pos].sum())
            den = (weights_real.sum())
            exp_frac = float(num / (den + 1e-12))
            print(f"[real sampler] hard_neg target={frac_hard:.3f}, expectedâ‰ˆ{exp_frac:.3f}, "
                  f"n_hard={n_hard}, n_non={n_non}")
        else:
            print("[real sampler] No hard-negatives found in the train subset for the provided CSV.")

    sampler_real = WeightedRandomSampler(
        torch.from_numpy(weights_real), num_samples=len(train_subset), replacement=True
    )
    train_loader_real = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler_real,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=True,
    )

    # ---- SYNTH training loader (optional) ----
    train_loader_synth = None
    if synth_root is not None:
        synth_img_dir = os.path.join(synth_root, 'images')
        synth_lbl_dir = os.path.join(synth_root, 'labels1d')
        synth_ds = M4DSAROilSpillSynthDataset(synth_img_dir, synth_lbl_dir, transform=augmented_tf)

        if synth_frac_targets is None:
            # >>> SIMPLE RANDOM SAMPLING <<<
            train_loader_synth = DataLoader(
                synth_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                drop_last=True,
            )
            # ---- NEW: sanity prints
            tags = np.array(getattr(synth_ds, "tags", []))
            if tags.size:
                uniq, cnt = np.unique(tags, return_counts=True)
                comp = {u: int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
            else:
                comp = {}
            print(f"[synth] random sampling (no class targets); N={len(synth_ds)}; tag_counts={comp}")
            # show two example pairs to confirm filename identity mapping
            if len(synth_ds) >= 2:
                print(f"[synth] sample[0] img={synth_ds.image_files[0]}  lbl={synth_ds.label_files[0]}  tag={synth_ds.tags[0]}")
                print(f"[synth] sample[1] img={synth_ds.image_files[1]}  lbl={synth_ds.label_files[1]}  tag={synth_ds.tags[1]}")
        else:
            # (keep your existing weighted-sampler code here for when targets are provided)
            tags = np.array(synth_ds.tags)
            counts = {t: max(1, np.sum(tags == t)) for t in set(tags)}
            active = {t: synth_frac_targets.get(t, 0.0) for t in counts.keys()}
            total = sum(active.values())
            if total <= 0:
                active = {t: 1.0 for t in counts.keys()}
                total = float(len(active))
            active = {t: v/total for t,v in active.items()}
            w_by_tag = {t: (active[t] / counts[t]) for t in counts.keys()}
            weights_synth = np.array([w_by_tag[tt] for tt in tags], dtype=np.float32)
            sampler_synth = WeightedRandomSampler(
                torch.from_numpy(weights_synth), num_samples=len(synth_ds), replacement=True
            )
            train_loader_synth = DataLoader(
                synth_ds, batch_size=batch_size, sampler=sampler_synth,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=(num_workers > 0), drop_last=True,
            )
            comp = {t: int(np.sum(tags == t)) for t in counts.keys()}
            print(f"[synth] counts={comp}  targets={active}  (N={len(synth_ds)})")

    print(f"Dataset: train(real) {len(train_idx)}, valid {len(val_idx)}")
    if n_hard_found:
        print(f"Hard-negatives in real train subset: {n_hard_found} / {len(train_subset)} (targetâ‰ˆ{frac_hard:.2f})")

    return train_loader_real, train_loader_synth, valid_loader
