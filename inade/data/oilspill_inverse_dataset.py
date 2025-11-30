import os
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class OilspillInverseDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        phase = opt.phase  # normalmente "train"

        # SAR en carpeta images/, máscaras 1D en labels/
        self.dir_label = os.path.join(self.root, phase, 'images')   # <- imagen SAR
        self.dir_image = os.path.join(self.root, phase, 'labels')   # <- máscara semántica

        self.label_paths = sorted(make_dataset(self.dir_label))
        self.image_paths = sorted(make_dataset(self.dir_image))

        self.dataset_size = len(self.label_paths)

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        image_path = self.image_paths[index]

        # SAR como input
        image_img = Image.open(label_path).convert('RGB')

        # Máscara 1D como target
        label_img = Image.open(image_path).convert('L')

        params = get_params(self.opt, image_img.size)
        transform_image = get_transform(self.opt, params)  # SAR
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)  # Máscara

        image_tensor = transform_image(image_img)  # [3, H, W]
        label_tensor = transform_label(label_img) * 255.0  # [1, H, W]
        label_tensor = label_tensor.long()

        # Validación de rango
        if label_tensor.max() >= self.opt.label_nc:
            raise ValueError(f"❌ Valor fuera de rango en la máscara: {label_tensor.max()} (label_nc={self.opt.label_nc})")

        return {
            'label': label_tensor,     # <- máscara 1D
            'image': image_tensor,     # <- imagen SAR RGB
            'path': label_path
        }

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'OilspillInverseDataset'