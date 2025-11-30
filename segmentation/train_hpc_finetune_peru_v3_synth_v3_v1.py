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
from train_loop_finetune_peru_final_v3_synth_v3 import batch_train
import os
# Define the destination directory in your Google Drive
drive_destination_dir = '/data/ajuarez/Results_multiple_oil_spill_final_v3'

# Create the directory if it doesn't exist


if not os.path.exists(drive_destination_dir):
    os.makedirs(drive_destination_dir)
    print(f"Created directory: {drive_destination_dir}")
else:
    print(f"Directory already exists: {drive_destination_dir}")

torch.set_num_threads(32)  # ajusta a tu gusto
class FLAGS:
    # Adjusted paths
    krest_dir = '/data/ajuarez/peru_split_8020_corregidas_v3_stride384'
    dir_model = os.path.join(drive_destination_dir, 'models')  # Save models directly to Google Drive
    dir_state = os.path.join(drive_destination_dir, 'states')  # Save optimizer states directly to Google Drive
    drive_destination_dir = '/data/ajuarez/Results_multiple_oil_spill_final_v3'
    pretrained = 1  # 1 for True, 0 for False
    random_state = 42
    which_optimizer = "adamw"  # or "adamw"
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 40
    batch_size = 50
    num_classes = 5  # Update if necessary
    which_model = None
    patience = 15
    num_workers = 30
    continue_training = True
    # Removed redundant dir_model and other paths
    file_model_weights = None  # Update this path if needed
    dir_save_preds = None  # Update this path if needed
    hardneg_csv = '/data/ajuarez/peru_split_8020_corregidas_v3_stride384/hard_negatives.csv'  # <- put it here
    frac_hard   = 0.10
    flat_weight = 1.5
    synth_scales = [0.25,0.5,1,1.5,2]   # add 1.0 if you also want the baseline
    real_steps = 2       # how many REAL mini-batches per cycle
    synth_steps = 1      # how many SYNTH mini-batches per cycle
    save_every   = 2             # save .pt/.state every N epochs


def main():
    print(">>> Entrando a main()")
    print("Verificando que hay modelos a entrenar...")
    baseline_state_dir = os.path.join(
        FLAGS.dir_state,  # e.g., /data/ajuarez/Results_multiple_oil_spill_final_v3/states
        "swin_unet_tiny",  # or FLAGS.which_model if you loop multiple
        "full_finetune_baseline3" # baseline mode name you used
    )

    # 2) define experiments
    experiments = [
        {
            "name": "morph00_902_steps1",
            "synth_root": os.path.join(FLAGS.krest_dir, "train_synth_nomorph_902")
        },
        {
            "name": "morph50_902_multi",
            "synth_root": os.path.join(FLAGS.krest_dir, "train_synth_50morph_902")
        },
        {
            "name": "morph100_902_multi",
            "synth_root": os.path.join(FLAGS.krest_dir, "train_synth_100morph_902")
        },

    ]
    # 3) run them sequentially (same starting weights, independent outputs)
    FLAGS.which_model = "swin_unet_tiny"
    for exp in experiments:
        for scale in FLAGS.synth_scales:
            FLAGS.synth_scale = float(scale)  # used later inside batch_train
            # make output dirs unique per (exp, scale)
            FLAGS.results_group   = "synth_results"
            FLAGS.experiment_name = f"{exp['name']}_scale{int(scale*100)}"
            FLAGS.init_from_dir = baseline_state_dir
            FLAGS.reset_epoch_counter = True
            FLAGS.synth_root      = exp["synth_root"]  # this is what determines how many synth batches exist

            print(f"\n=== {FLAGS.which_model} | EXP: {exp['name']} | scale={scale} ===")
            batch_train(FLAGS, mode='full_finetune_baseline3')
if __name__ == "__main__":
    main()