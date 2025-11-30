import os
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class OilspillDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        phase = opt.phase  # normalmente "train"

        self.dir_A = os.path.join(self.root, phase, 'labels_1D')
        self.dir_B = os.path.join(self.root, phase, 'images')
        self.dir_I = os.path.join(self.root, phase, 'inst')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.I_paths = sorted(make_dataset(self.dir_I))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        I_path = self.I_paths[index]

        A = Image.open(A_path).convert('L')  # Máscara semántica (1 canal)
        B = Image.open(B_path).convert('L').convert('RGB')
        I = Image.open(I_path).convert('L')  # Mapa de instancia

        # 💡 Crear los parámetros para transformaciones de tamaño/corte
        params = get_params(self.opt, A.size)

        # Aplicar las transformaciones a cada imagen
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_B = get_transform(self.opt, params)
        transform_I = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        A_tensor = transform_A(A).long() 
        B_tensor = transform_B(B)
        I_tensor = transform_I(I).long() 

        return {
            'label': A_tensor,
            'image': B_tensor,
            'instance': I_tensor,
            'path': A_path
        }

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'OilspillDataset'