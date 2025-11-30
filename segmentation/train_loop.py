from logger_utils import CSVWriter, write_dict_to_json
from seg_models import ResNet34DeepLabV3Plus,ResNet50DeepLabV3Plus,ResNet101DeepLabV3Plus
from seg_models import EfficientNetMDeepLabV3, ResNet34UNet, ResNet34UNetWithASPP,TransUNet,SwinUNetTiny
from metrics import compute_mean_pixel_acc, compute_mean_IOU, compute_class_IOU
from logger_utils import CSVWriter, write_dict_to_json
from dataset import get_dataloaders_for_training
import torchvision.transforms as transforms
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
import torch
import csv
import json
import glob
import torch.nn.functional as F
import numpy as np
import re
#all metrics
#validation & train and polinomial loop
from torch.optim.lr_scheduler import _LRScheduler

def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    dice = 0
    for i in range(num_classes):
        pred_i = pred[:, i, :, :]
        target_i = (target == i).float()
        intersection = (pred_i * target_i).sum(dim=(1,2))
        union = pred_i.sum(dim=(1,2)) + target_i.sum(dim=(1,2))
        dice += (2. * intersection + smooth) / (union + smooth)
    dice = dice / num_classes
    return 1 - dice.mean()

def get_b16_config(num_classes):
    config = {
        'hidden_size': 768,
        'transformer': {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        },
        'classifier': 'seg',
        'representation_size': None,
        'decoder_channels': [256, 128, 64, 16],
        'n_classes': num_classes,
        'activation': 'softmax',
    }
    return config

def get_r50_b16_config(num_classes):
    config = get_b16_config(num_classes)
    config['patches'] = {'grid': (16, 16)}
    config['resnet'] = {
        'num_layers': (3, 4, 9),  # ResNet-50 layers (3 blocks)
        'width_factor': 1,        # Set to 1
    }
    config['n_skip'] = 3
    config['skip_channels'] =  [1024, 512, 256]
    config['decoder_channels'] = [256, 128, 64, 16]
    config['img_size'] = 512  # Adjusted to be divisible by 32 and 16
    config['n_classes'] = num_classes
    config['activation'] = 'softmax'
    return config


class PolynomialLR(_LRScheduler):
    """
    PolynomialLR class for the polynomial learning rate scheduler

    ----------
    Attributes
    ----------
    optimizer : object
        object of type optimizer
    max_epochs : int
        maximum number of epochs for which optimization needs to be run
    power : float
        the power term in the polynomial learning rate scheduler (default: 0.9)
    last_epoch : int
        last epoch in the optimization (default: -1)
    min_lr : float
        minimum value for the learning rate (default: 1e-6)
    """

    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_epochs = max_epochs
        self.min_lr = min_lr  # avoid zero lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            max(
                base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power,
                self.min_lr,
            )
            for base_lr in self.base_lrs
        ]


def validation_loop(dataset_loader, model, ce_loss, device):
    model.eval()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    valid_loss, valid_acc, valid_IOU = 0, 0, 0

    with torch.no_grad():
        for image, label in dataset_loader:
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            pred_logits = model(image)
            valid_loss += ce_loss(pred_logits, label)+  dice_loss(pred_logits, label)

            pred_probs = F.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)

            valid_acc += compute_mean_pixel_acc(label, pred_label)
            valid_IOU += compute_mean_IOU(label, pred_label)

    valid_loss /= num_batches
    valid_acc /= num_batches
    valid_IOU /= num_batches
    return valid_loss, valid_acc, valid_IOU


def train_loop(dataset_loader, model, ce_loss, optimizer, device):
    """
    ---------
    Arguments
    ---------
    dataset_loader : object
        object of type dataloader
    model : object
        object of type model
    ce_loss : object
        object of type cross entropy loss
    optimizer : object
        object of type optimizer
    device : str
        device on which training needs to be run

    -------
    Returns
    -------
    train_loss : torch float
        mean loss for the training set
    """
    model.train()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    train_loss = 0

    for image, label in dataset_loader:
        image = image.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        optimizer.zero_grad()

        pred_logits = model(image)
        loss = ce_loss(pred_logits, label)+ dice_loss(pred_logits, label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss








class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

import time
import os
import shutil

class CSVWriter:
    def __init__(self, file_name, column_names, append=False):
        self.file_name = file_name
        self.append = append
        self.file_exists = os.path.exists(file_name)
        self.file = open(file_name, 'a' if append else 'w', newline='')
        self.writer = csv.writer(self.file)
        if not self.append or not self.file_exists:
            self.writer.writerow(column_names)  # Only write header if new
            self.file.flush()
            os.fsync(self.file.fileno())

    def write_row(self, row):
        self.writer.writerow(row)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()
def _epoch_from_path(path, model_name):
    # escapamos el nombre para que los caracteres especiales no cuenten
    m = re.search(rf"{re.escape(model_name)}_(\d+)\.state$", path)
    return int(m.group(1)) if m else None
def _find_last_state(dir_state, model_name):
    patt = os.path.join(dir_state, f"oil_spill_seg_{model_name}_*.state")
    candidates = [
        (ep, p) for p in glob.glob(patt)
        if (ep := _epoch_from_path(p, model_name)) is not None
    ]
    return max(candidates, default=(0, None), key=lambda t: t[0])

def _update_progress(root_dir, model_name, epoch):
    """Guarda/actualiza progress.json con la epoca alcanzada."""
    pfile = os.path.join(root_dir, "progress.json")
    prog = {}
    if os.path.exists(pfile):
        with open(pfile, "r") as fp:
            prog = json.load(fp)
    prog[model_name] = epoch
    with open(pfile, "w") as fp:
        json.dump(prog, fp, indent=2)

def _prune_states(dir_state, model_name, keep_last=3):
    patt = os.path.join(dir_state, f"oil_spill_seg_{model_name}_*.state")
    states = sorted(
        [(ep, p) for p in glob.glob(patt)
         if (ep := _epoch_from_path(p, model_name)) is not None],
        key=lambda t: t[0]
    )
    for _, f in states[:-keep_last]:
        try:
            os.remove(f)
        except OSError:
            pass



def batch_train(FLAGS):
    # Local directory to save models and metrics temporarily
    local_dir = os.path.join('/tmp', 'models', FLAGS.which_model)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
        print(f"Created local directory: {local_dir}")

    # Directory on Google Drive
    dir_path = os.path.join(FLAGS.dir_model, FLAGS.which_model)
    dir_state = os.path.join(FLAGS.dir_state, FLAGS.which_model)
    drive_destination_dir = FLAGS.drive_destination_dir
    os.makedirs(dir_state, exist_ok=True)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory on Google Drive: {dir_path}")
    # Initialize CSVWriter with local file path
    '''
    drive_metrics = os.path.join(dir_path, "train_metrics.csv")
    if FLAGS.continue_training and os.path.exists(drive_metrics):
        shutil.copyfile(drive_metrics, metrics_file_local)
    append_metrics = os.path.exists(metrics_file_local)
    
    csv_writer = CSVWriter(
        file_name=metrics_file_local,
        column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_IOU"],
        append=append_metrics)'''

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    krest_dir = FLAGS.krest_dir  # Make sure to set this in your FLAGS

    train_dataset_loader, valid_dataset_loader = get_dataloaders_for_training(
        krest_dir=krest_dir,
        batch_size=FLAGS.batch_size,
        random_state=FLAGS.random_state,
        num_workers=FLAGS.num_workers
    )

    # Model selection
    if FLAGS.which_model == "resnet_34_deeplab_v3+":
        oil_spill_seg_model = ResNet34DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )

    elif FLAGS.which_model == "transunet":
        #from transunet.configs import get_r50_b16_config
        config = get_r50_b16_config(FLAGS.num_classes)
        oil_spill_seg_model = TransUNet(
            config=config,
            img_size=config['img_size'],
            num_classes=config['n_classes'],
            vis=False
        )
        if bool(FLAGS.pretrained):
            print("Loading pre-trained weights for TransUNet...")
            # Load pre-trained weights
            pretrained_weights = np.load('/home/ajuarez/imagenet21k_R50+ViT-B_16.npz')
            oil_spill_seg_model.load_from(weights=pretrained_weights)
    elif FLAGS.which_model == "resnet_50_deeplab_v3+":
        oil_spill_seg_model = ResNet50DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "resnet_101_deeplab_v3+":
        oil_spill_seg_model = ResNet101DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "efficientnet_v2_m_deeplab_v3":
        oil_spill_seg_model = EfficientNetMDeepLabV3(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "resnet_34_unet":
      oil_spill_seg_model = ResNet34UNet(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
            )
    elif FLAGS.which_model  == "resnet_34_unet_aspp":
        oil_spill_seg_model = ResNet34UNetWithASPP(num_classes=FLAGS.num_classes, pretrained = bool(FLAGS.pretrained)
        )
    elif FLAGS.which_model == "swin_unet_tiny":
        in_ch = getattr(FLAGS, "in_ch", 3)  # por defecto 1para kresti, sino lo que se defina en FLAGS
        oil_spill_seg_model = SwinUNetTiny(num_classes=FLAGS.num_classes, in_channels=in_ch, pretrained=bool(FLAGS.pretrained)
        )
    else:
        print("model not yet implemented, so exiting")
    oil_spill_seg_model.to(device)

        # Optimizer selection
    if FLAGS.which_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            oil_spill_seg_model.parameters(),
            lr=FLAGS.learning_rate,
            momentum=0.9,
            weight_decay=FLAGS.weight_decay,
        )
        lr_scheduler = PolynomialLR(
            optimizer,
            FLAGS.num_epochs + 1,
            power=0.9,
        )
    elif FLAGS.which_optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            oil_spill_seg_model.parameters(),
            lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',              # miramos valid_loss
            factor=0.3,              # baja LR ×0.3
            patience=3,              # si no mejora en 3 epochs
            verbose=True
        )
    
    
    start_epoch = 1                # default = fresh run
    ckpt = None                    # only defined if we really load one
    drive_metrics = os.path.join(dir_path, "train_metrics.csv")
    metrics_file_local = os.path.join(local_dir, "train_metrics.csv")
    _, last_state = _find_last_state(dir_state, FLAGS.which_model)

    if FLAGS.continue_training and last_state is not None:
        # --------- ? resume from full state
        ckpt = torch.load(last_state, map_location=device)
        oil_spill_seg_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        lr_scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"? Resuming from .state (epoch {ckpt['epoch']})")
        if os.path.exists(drive_metrics):
            shutil.copyfile(drive_metrics, metrics_file_local)
            print(f"Copied previous metrics from {drive_metrics}")
        append_metrics = True

    elif FLAGS.continue_training:
        # --------- ? try the bare .pt checkpoint
        pt_files = glob.glob(
            os.path.join(dir_path, f"oil_spill_seg_{FLAGS.which_model}_*.pt"))
        
        pt_matches = [
            (int(m.group(1)), f)
            for f in pt_files
            if (m := re.search(rf"{re.escape(FLAGS.which_model)}_(\d+)\.pt$", f))
        ]

        if pt_matches:
            last_epoch_pt, last_pt = max(pt_matches, key=lambda t: t[0])
            oil_spill_seg_model.load_state_dict(
                torch.load(last_pt, map_location=device))
            start_epoch = last_epoch_pt + 1
            print(f"? Resuming from .pt (epoch {last_epoch_pt}, fresh optimiser)")
            if os.path.exists(drive_metrics):
                shutil.copyfile(drive_metrics, metrics_file_local)
                print(f"Copied previous metrics from {drive_metrics}")
            append_metrics = True
        else:
            print("? No valid .pt checkpoint found starting from scratch")
            print("? Starting from epoch 1 (fresh training run)")
            append_metrics = False
    else:
        # --------- ? fresh run requested
        print("? User requested fresh training run")
        append_metrics = False
    
    csv_writer = CSVWriter(
    file_name=metrics_file_local,
    column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_IOU"],
    append=append_metrics)
    #class_weights = torch.tensor([0.003202448	,0.19367078,	0.03739316,	0.085356423	,4.680377189]).to(device)
    class_weights = torch.tensor([0.089950749,0.699513043,	0.307368848,	3.438778419	,0.464388941]).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"\nTraining oil spill segmentation model: {FLAGS.which_model}\n")


    # Save training parameters locally and copy to Google Drive
    serializable_flags = {k: v for k, v in vars(FLAGS).items() if isinstance(v, (int, float, str, list, dict))}
    params_file_local = os.path.join(local_dir, "params.json")
    write_dict_to_json(params_file_local, serializable_flags)
    params_file_drive = os.path.join(dir_path, "params.json")
    shutil.copyfile(params_file_local, params_file_drive)

    early_stopping = EarlyStopping(patience=FLAGS.patience, verbose=True, path=os.path.join(dir_path, 'checkpoint.pt'))

    for epoch in range(start_epoch, FLAGS.num_epochs + 1):
        t_1 = time.time()
        train_loss = train_loop(
            train_dataset_loader, oil_spill_seg_model, ce_loss, optimizer, device
        )
        t_2 = time.time()
        print("-" * 100)
        print(
            f"Epoch : {epoch}/{FLAGS.num_epochs}, Time: {(t_2 - t_1):.2f} sec., Train Loss: {train_loss:.5f}"
        )
        valid_loss, valid_acc, valid_IOU = validation_loop(
            valid_dataset_loader, oil_spill_seg_model, ce_loss, device
        )
        print(
            f"Validation Loss: {valid_loss:.5f}, Validation Accuracy: {valid_acc:.5f}, Validation IOU: {valid_IOU:.5f}"
        )
	        # Early stopping (if implemented)
        # early_stopping(valid_loss)
        early_stopping(valid_loss, oil_spill_seg_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Write metrics to CSV and flush immediately
        csv_writer.write_row(
            [
                epoch,
                round(train_loss.item(), 5),
                round(valid_loss.item(), 5),
                round(valid_acc.item(), 5),
                round(valid_IOU.item(), 5),
            ]
        )
        # Flush and sync the CSV file
        csv_writer.file.flush()
        os.fsync(csv_writer.file.fileno())

        # Copy the metrics file to Google Drive
        metrics_file_local = os.path.join(local_dir, "train_metrics.csv")

        shutil.copyfile(metrics_file_local, drive_metrics)

        # Wait until the metrics file is confirmed to be in Google Drive
        while not os.path.exists(drive_metrics ):
            print("Waiting for the metrics file to be copied to Google Drive...")
            time.sleep(1)
        print(f"Metrics file saved to {drive_metrics }")

        # Save model checkpoint locally

        local_model_checkpoint_path = os.path.join(local_dir, f"oil_spill_seg_{FLAGS.which_model}_{epoch}.pt")
        with open(local_model_checkpoint_path, 'wb') as f:
            torch.save(oil_spill_seg_model.state_dict(), f)
            f.flush()
            os.fsync(f.fileno())

        # Copy the model checkpoint to Google Drive
        drive_model_checkpoint_path = os.path.join(dir_path, f"oil_spill_seg_{FLAGS.which_model}_{epoch}.pt")
        shutil.copyfile(local_model_checkpoint_path, drive_model_checkpoint_path)
        
        
        state_path_local  = os.path.join(
            local_dir, f"oil_spill_seg_{FLAGS.which_model}_{epoch}.state")
        torch.save({
            'epoch': epoch,
            'model_state': oil_spill_seg_model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': lr_scheduler.state_dict()
        }, state_path_local)
        shutil.copyfile(state_path_local,
            os.path.join(dir_state,
                        f"oil_spill_seg_{FLAGS.which_model}_{epoch}.state"))
        _prune_states(dir_state, FLAGS.which_model)  
        # 3) actualizar progreso global
        _update_progress(drive_destination_dir, FLAGS.which_model, epoch)


        # Wait until the model file is confirmed to be in Google Drive
        while not os.path.exists(drive_model_checkpoint_path):
            print("Waiting for the model file to be copied to Google Drive...")
            time.sleep(1)
        print(f"Model checkpoint saved to {drive_model_checkpoint_path}")

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # must pass the validation metric that the scheduler will monitor
            lr_scheduler.step(valid_loss)
        else:
            lr_scheduler.step()
        



    print("Training complete!")
    csv_writer.close()
    # Copy the final metrics file to Google Drive
    shutil.copyfile(os.path.join(local_dir, "train_metrics.csv"), os.path.join(dir_path, "train_metrics.csv"))
    return