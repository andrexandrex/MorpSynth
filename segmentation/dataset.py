import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,ConcatDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_trans
from torchvision.transforms import InterpolationMode

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
import albumentations as A
from albumentations.pytorch import ToTensorV2

mean = [0.4854, 0.4854, 0.4854]
std  = [0.1782, 0.1782, 0.1782]

augmented_tf = A.Compose([
    A.Resize(512, 512),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ], p=1.0),
    A.OneOf([
    A.RandomGamma(gamma_limit=(80,120), p=0.5),
    A.GaussNoise(std_range=(0.05,0.15), p=0.5),
    ], p=1.0),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

plain_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

valid_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])


class M4DSAROilSpillDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform  = transform
        self.files = sorted([
            f for f in os.listdir(images_dir)
            if f.startswith('img_') and f.endswith('.png')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_fn = self.files[idx]
        img_path = os.path.join(self.images_dir, img_fn)
        lbl_path = os.path.join(self.labels_dir, img_fn.replace('.png','_label_1D.png'))

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(lbl_path).convert('L'))

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']
        else:
            image = torch.as_tensor(image.transpose(2,0,1), dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.long)

        return image, label


def get_dataloaders_for_training(
    krest_dir,
    batch_size,
    num_workers,
    random_state=None
):
    train_img_dir  = os.path.join(krest_dir, 'train', 'images')
    train_lbl_dir  = os.path.join(krest_dir, 'train', 'labels_1D')

    # Dataset original
    plain_ds = M4DSAROilSpillDataset(train_img_dir, train_lbl_dir, transform=plain_tf)
    # Dataset augmentado
    aug_ds   = M4DSAROilSpillDataset(train_img_dir, train_lbl_dir, transform=augmented_tf)
    # Dataset combinado (duplica los datos!)
    combined_ds = ConcatDataset([plain_ds, aug_ds])

    # Validaci�n con dataset normal (sin duplicar!)
    valid_ds = M4DSAROilSpillDataset(train_img_dir, train_lbl_dir, transform=valid_tf)

    # Split para validaci�n
    n = len(plain_ds)
    idx = np.arange(n)
    split = int(0.05 * n)
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[split:], idx[:split]

    # Samplers
    train_sampler = SubsetRandomSampler(
        list(train_idx) + [i + n for i in train_idx]  # duplicado!
    )
    val_sampler = SubsetRandomSampler(val_idx)

    # DataLoaders
    train_loader = DataLoader(
        combined_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )

    print(f"Dataset: train {len(train_idx)*2}, valid {len(val_idx)}")
    return train_loader, valid_loader