import copy
import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
import torchvision.transforms.functional as F_trans
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torchvision.transforms import InterpolationMode
import ee
import pandas as pd
import openpyxl
import datetime
import os
from PIL import Image
import numpy as np
import json
import os
import json
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import time
import csv
from train_loop import batch_train
import os
# Define the destination directory in your Google Drive
drive_destination_dir = '/home/ajuarez/Results_multiple_oil_spill_final_v3'

# Create the directory if it doesn't exist


if not os.path.exists(drive_destination_dir):
    os.makedirs(drive_destination_dir)
    print(f"Created directory: {drive_destination_dir}")
else:
    print(f"Directory already exists: {drive_destination_dir}")

torch.set_num_threads(80)  # ajusta a tu gusto
class FLAGS:
    # Adjusted paths
    krest_dir = '/home/ajuarez/data/selected_sampling_final_less_bg_addmoresea'
    dir_model = os.path.join(drive_destination_dir, 'models')  # Save models directly to Google Drive
    dir_state = os.path.join(drive_destination_dir, 'states')  # Save optimizer states directly to Google Drive
    drive_destination_dir = '/home/ajuarez/Results_multiple_oil_spill_final_v3'
    pretrained = 1  # 1 for True, 0 for False
    random_state = 42
    which_optimizer = "adamw"  # or "adamw"
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 60
    batch_size = 32
    num_classes = 5  # Update if necessary
    which_model = None
    patience = 10
    num_workers = 79
    continue_training = True
    # Removed redundant dir_model and other paths
    file_model_weights = None  # Update this path if needed
    dir_save_preds = None  # Update this path if needed


def main():
	print(">>> Entrando a main()")
	print("Verificando que hay modelos a entrenar...")
	model_names = [
        "swin_unet_tiny",
    ]
	for model_name in model_names:
        	print(f"\nStarting training for model: {model_name}\n")
        	FLAGS.which_model = model_name  # Set the current model
        	batch_train(FLAGS)
        	print(f"\nTraining completed for model: {model_name}\n")

if __name__ == "__main__":
    main()