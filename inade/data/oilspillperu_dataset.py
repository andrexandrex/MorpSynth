import os, re
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

# regex: build same key for images (*_patch_#) and masks (*_mask1d_#)
IMG_RE  = re.compile(r"^(?P<prefix>.+?)_patch_(?P<idx>\d+)\.(?:png|tif|tiff)$", re.IGNORECASE)
MASK_RE = re.compile(r"^(?P<prefix>.+?)_mask1d_(?P<idx>\d+)\.png$", re.IGNORECASE)

def index_by_key(folder, pat):
    out = {}
    for p in make_dataset(folder):
        name = os.path.basename(p)
        m = pat.match(name)
        if m:
            key = f"{m.group('prefix')}_{m.group('idx')}"
            out[key] = p
    return out

class OilspillperuDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        phase = opt.phase  # "train" | "val" | "test"

        self.dir_A = os.path.join(self.root, phase, 'labels1d')  # 0..4 IDs, 8-bit PNG
        self.dir_B = os.path.join(self.root, phase, 'images')  # real images (*.tif/png), *_patch_#
        self.dir_I = os.path.join(self.root, phase, 'inst')    # instances, 8-bit PNG

        A = index_by_key(self.dir_A, MASK_RE)
        B = index_by_key(self.dir_B, IMG_RE)
        I = index_by_key(self.dir_I, MASK_RE)

        keys = sorted(set(A) & set(B) & set(I))
        # hard fail on mispairs
        if not (len(keys) == len(A) == len(B) == len(I)):
            missing = []
            for k in sorted(set(A) | set(B) | set(I)):
                present = [src for src, d in [('label',A), ('image',B), ('inst',I)] if k in d]
                if len(present) != 3:
                    missing.append(f"{k}: present -> {present}")
            raise RuntimeError("Unpaired files detected:\n" + "\n".join(missing[:100]))

        self.A_paths = [A[k] for k in keys]
        self.B_paths = [B[k] for k in keys]
        self.I_paths = [I[k] for k in keys]
        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        I_path = self.I_paths[index]

        # labels: already 0..4 IDs, 8-bit
        A = Image.open(A_path).convert('L')
        # images: if grayscale .tif, expand to 3ch
        B = Image.open(B_path)
        if B.mode != 'RGB':
            B = B.convert('L').convert('RGB')
        # instances: 8-bit
        I = Image.open(I_path).convert('L')

        params = get_params(self.opt, A.size)

        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_B = get_transform(self.opt, params)
        transform_I = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        # keep integer IDs
        A_tensor = (transform_A(A) * 255.0).long()   # -> {0..4}
        B_tensor = transform_B(B)                    # float normalized image
        I_tensor = (transform_I(I) * 255.0).long()   # -> instance IDs

        return {
            'label': A_tensor,
            'image': B_tensor,
            'instance': I_tensor,
            'path': A_path
        }

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'OilspillperuDataset'
