from logger_utils import CSVWriter, write_dict_to_json
from seg_models import ResNet34DeepLabV3Plus,ResNet50DeepLabV3Plus,ResNet101DeepLabV3Plus
from seg_models import EfficientNetMDeepLabV3, ResNet34UNet, ResNet34UNetWithASPP,TransUNet
from metrics import compute_mean_pixel_acc, compute_mean_IOU, compute_class_IOU
from logger_utils import CSVWriter, write_dict_to_json
from dataset_peru_train_v3_synth import get_dataloaders_for_training
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


def _set_bn_eval(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()



def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
    # logits: [B,C,H,W], target: [B,H,W]
    p = F.softmax(logits, dim=1)
    B, C, H, W = p.shape
    loss = 0.0
    for c in range(C):
        pc = p[:, c]
        tc = (target == c).float()
        TP = (pc * tc).sum(dim=(1,2))
        FP = (pc * (1.0 - tc)).sum(dim=(1,2))
        FN = ((1.0 - pc) * tc).sum(dim=(1,2))
        TI = (TP) / (TP + alpha*FP + beta*FN + smooth)
        FTL = torch.pow(1.0 - TI, gamma)
        loss += FTL.mean()
    return loss / C
def confusion_penalty(logits, target, from_cls=2, to_cls=1, lam=0.5, gamma=2.0):
    # penalize prob of 'to_cls' where GT is 'from_cls'
    with torch.no_grad():
        mask = (target == from_cls)
    p_to = torch.softmax(logits, dim=1)[:, to_cls, ...]
    # focal-like: emphasize confident mistakes
    penalty = (p_to ** gamma)[mask].mean() if mask.any() else logits.new_tensor(0.0)
    return lam * penalty


def effective_num_weights_from_counts(counts, beta=None):
    import numpy as np
    counts = np.asarray(counts, dtype=np.float64)
    max_n = counts.max()
    if beta is None:
        beta = (max_n - 1.0) / max_n   # ~1 - 1/max_n (paper’s suggestion)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-12)
    w = w / np.mean(w)                 # normalize to mean=1
    return w


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


def validation_loop(dataset_loader, model, loss_fn, device, num_classes=None, ignore_index=None):
    """
    Returns:
        valid_loss      : scalar tensor
        pixel_acc       : float
        mean_iou        : float (nan-safe)
        per_class_iou   : np.ndarray shape [C]
    """
    model.eval()
    num_batches = len(dataset_loader)

    # lazy infer num_classes if not provided
    if num_classes is None:
        # quick probe (safe: we won't keep these tensors)
        for image, _label in dataset_loader:
            with torch.no_grad():
                _logits = model(image.to(device, dtype=torch.float))
            num_classes = _logits.shape[1]
            del _logits
            break
        if num_classes is None:
            raise ValueError("num_classes could not be inferred; please pass it explicitly.")

    # Accumulators
    valid_loss = 0.0
    intersect = torch.zeros(num_classes, dtype=torch.double, device=device)
    union     = torch.zeros(num_classes, dtype=torch.double, device=device)
    correct   = 0
    total_pix = 0

    with torch.no_grad():
        for image, label in dataset_loader:
            image = image.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            logits = model(image)
            valid_loss += loss_fn(logits, label)

            # predictions
            preds = logits.argmax(dim=1)  # [B,H,W]

            # optionally ignore pixels
            if ignore_index is not None:
                valid_mask = (label != ignore_index)
                # pixel accuracy
                correct   += (preds.eq(label) & valid_mask).sum().item()
                total_pix += valid_mask.sum().item()
            else:
                correct   += preds.eq(label).sum().item()
                total_pix += label.numel()

            # per-class IoU accumulators
            # We'll exclude ignore_index from unions/intersections implicitly by masking.
            for cls in range(num_classes):
                p = (preds == cls)
                g = (label == cls)
                if ignore_index is not None:
                    p = p & valid_mask
                    g = g & valid_mask
                intersect[cls] += torch.logical_and(p, g).sum()
                union[cls]     += torch.logical_or (p, g).sum()

    # averages
    valid_loss = valid_loss / num_batches

    pixel_acc = float(correct) / (float(total_pix) + 1e-9)

    # IoU per class (nan if union==0)
    union_safe = union + 1e-9
    per_class_iou = (intersect / union_safe)  # tensor [C]
    per_class_iou_np = per_class_iou.clamp(min=0.0, max=1.0).cpu().numpy()

    # mean IoU ignoring classes with union==0 (no pixels present across val set)
    valid_classes = (union > 0)
    if valid_classes.any():
        mean_iou = float((per_class_iou[valid_classes]).mean().item())
    else:
        mean_iou = float('nan')

    return valid_loss, pixel_acc, mean_iou, per_class_iou_np



from itertools import cycle

def train_loop_mixed_ratio(
    train_loader_real,
    train_loader_synth,
    model,
    loss_fn,
    optimizer,
    device,
    *,
    real_steps: int = 2,
    synth_steps: int = 1,
    synth_loss_scale: float = 1.0,
):
    """
    Interleave real and synthetic mini-batches in the ratio real_steps : synth_steps.
    Anchors an epoch to ~one pass over the REAL loader (len(train_loader_real)).

    Returns:
        mean_loss (torch.float32) over all executed mini-batches (real + synth)
    """
    model.train()
    loss_sum = 0.0
    n_steps  = 0

    it_real  = cycle(train_loader_real)
    it_synth = cycle(train_loader_synth) if train_loader_synth is not None else None

    # anchor length to real loader so each epoch sees ~one pass of real
    steps = len(train_loader_real)

    for _ in range(steps):
        # ---- real mini-batches ----
        for _r in range(max(1, int(real_steps))):
            xr, yr = next(it_real)
            xr, yr = xr.to(device, dtype=torch.float), yr.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits_r = model(xr)
            loss_r   = loss_fn(logits_r, yr)
            loss_r.backward()
            optimizer.step()
            loss_sum += float(loss_r.detach().cpu())
            n_steps  += 1

        # ---- synthetic mini-batches (optionally; also freeze BN stats here) ----
        if it_synth is not None and synth_steps > 0:
            model.apply(_set_bn_eval)  # keep BN running stats from drifting on synth
            for _s in range(int(synth_steps)):
                batch = next(it_synth)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    xs, ys, _tags = batch
                else:
                    xs, ys = batch
                xs, ys = xs.to(device, dtype=torch.float), ys.to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits_s = model(xs)
                loss_s   = loss_fn(logits_s, ys) * float(synth_loss_scale)
                loss_s.backward()
                optimizer.step()
                loss_sum += float(loss_s.detach().cpu())
                n_steps  += 1

    return torch.tensor(loss_sum / max(1, n_steps), dtype=torch.float32)



def train_loop(dataset_loader, model, loss_fn, optimizer, device):
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
        loss = loss_fn(pred_logits, label) 


        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss




def freeze_encoder(model, freeze=True):
    for name, p in model.named_parameters():
        # adjust 'encoder' substring to match your implementation
        if any(k in name for k in ['backbone', 'encoder', 'vit']):
            p.requires_grad = not freeze




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

def _prune_states(dir_state, model_name,mode, keep_last=3):
    patt = os.path.join(dir_state, f"ft_{mode}_oil_spill_seg_{model_name}_*.state")
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



def batch_train(FLAGS, mode:str):
    assert mode in ('freeze_backbone_baseline', 'full_finetune_baseline3')
    suffix = mode

    # === output roots: allow branching into synth_results/<experiment_name>/ ===
    # base roots (as you already set)
    base_model_root = FLAGS.dir_model
    base_state_root = FLAGS.dir_state

    # add optional grouping
    subdir_parts = [FLAGS.which_model]
    if getattr(FLAGS, "results_group", None):
        subdir_parts += [FLAGS.results_group]
    if getattr(FLAGS, "experiment_name", None):
        subdir_parts += [FLAGS.experiment_name]
    subdir_parts += [suffix]  # keep your 'mode' suffix at the end

    # final dirs
    local_dir = os.path.join('/tmp', 'models', *subdir_parts)
    dir_path  = os.path.join(base_model_root, *subdir_parts)
    dir_state = os.path.join(base_state_root, *subdir_parts)

    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(dir_path,  exist_ok=True)
    os.makedirs(dir_state, exist_ok=True)
    print(f"[paths] local_dir={local_dir}\n        dir_path={dir_path}\n        dir_state={dir_state}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    krest_dir = FLAGS.krest_dir  # Make sure to set this in your FLAGS

    train_loader_real, train_loader_synth, valid_dataset_loader = get_dataloaders_for_training(
        krest_dir=krest_dir,                         # PERU root (as you requested)
        batch_size=FLAGS.batch_size,
        random_state=FLAGS.random_state,
        num_workers=FLAGS.num_workers,
        hardneg_csv=getattr(FLAGS, 'hardneg_csv', None),
        frac_hard=getattr(FLAGS, 'frac_hard', 0.30),
        flat_weight=getattr(FLAGS, 'flat_weight', 3.0),
        synth_root=getattr(FLAGS, 'synth_root', None)          # <-- use per-exp mix
    )


    # Model selection
    if FLAGS.which_model == "resnet_34_deeplab_v3+":
        oil_spill_seg_model = ResNet34DeepLabV3Plus(
            num_classes=FLAGS.num_classes, pretrained=bool(FLAGS.pretrained)
        )

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
    else:
        print("model not yet implemented, so exiting")
    oil_spill_seg_model.to(device)
    if mode == 'freeze_backbone_baseline':
        freeze_encoder(oil_spill_seg_model, freeze=True)
    else:  # full_finetune
        freeze_encoder(oil_spill_seg_model, freeze=False)
    def freeze_bn(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters(): p.requires_grad = False

    if mode == 'freeze_backbone_baseline':
        oil_spill_seg_model.apply(freeze_bn)
    
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
        #trainable = [p for p in oil_spill_seg_model.parameters() if p.requires_grad]
        if mode == 'freeze_backbone_baseline':
            # decoder params get 1e-4, encoder (which are frozen anyway) get 0
            head_params = [p for n,p in oil_spill_seg_model.named_parameters() if p.requires_grad]
            optimizer   = torch.optim.AdamW([
                {'params': head_params, 'lr': FLAGS.learning_rate},
            ], weight_decay=FLAGS.weight_decay)
        else:
            # full fine-tune: small LR on backbone, larger on head
            encoder_params = [p for n,p in oil_spill_seg_model.named_parameters() if p.requires_grad and 'encoder' in n]
            head_params    = [p for n,p in oil_spill_seg_model.named_parameters() if p.requires_grad and 'encoder' not in n]
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': FLAGS.learning_rate * 0.1},
                {'params': head_params,    'lr': FLAGS.learning_rate},
            ], weight_decay=FLAGS.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',              # miramos valid_loss
            factor=0.3,              # baja LR  0.3
            patience=3,              # si no mejora en 3 epochs
            verbose=True
        )
    
    
    start_epoch = 1                # default = fresh run
    ckpt = None                    # only defined if we really load one
    original_state_dir = os.path.join(FLAGS.dir_state, FLAGS.which_model)
    drive_metrics = os.path.join(dir_path, "train_metrics_ft.csv")
    metrics_file_local = os.path.join(local_dir, "train_metrics_ft.csv")
    start_epoch = 1
    append_metrics = False
    if getattr(FLAGS, "init_from_dir", None):
        # e.g., FLAGS.init_from_dir = /home/.../states/<which_model>/full_finetune_baseline3
        def _find_last_state_in(dir_):
            last_ep, last_path = _find_last_state(dir_, FLAGS.which_model)
            # If your _find_last_state expects a flat dir, try a simpler finder:
            if last_path is None:
                # naive fallback: pick the latest *.state in this directory
                cand = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith(".state")]
                if cand:
                    cand.sort()
                    last_path = cand[-1]
            return last_path

        init_state = _find_last_state_in(FLAGS.init_from_dir)
        if init_state is None:
            print(f"[warn] No .state found in init_from_dir={FLAGS.init_from_dir}; starting from scratch.")
        else:
            print(f"[init] Loading model weights from: {init_state}")
            ckpt = torch.load(init_state, map_location=device)
            oil_spill_seg_model.load_state_dict(ckpt['model_state'])
            # fresh optimizer / scheduler, fresh epoch count
            start_epoch = 1
            append_metrics = False
    else:
        print("[init] No init_from_dir provided; starting from scratch (random init).")





    
    csv_writer = CSVWriter(
            file_name=metrics_file_local,
            column_names=[
                "epoch", "train_loss", "valid_loss", "valid_acc", "valid_mIoU",
                "c0_iou", "c1_iou", "c2_iou", "c3_iou", "c4_iou"
            ],
            append=append_metrics
        )
    freqs = np.array([392657571,  33335913,  24062560 ,   551480,  19940956], dtype=np.int64)
    w = effective_num_weights_from_counts(freqs)
    #class_weights = torch.tensor([0.003202448	,0.19367078,	0.03739316,	0.085356423	,4.680377189]).to(device)
    class_weights = torch.tensor(w, dtype=torch.float, device=device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    ALPHA_TV    = 0.65   # ↑ to penalize FP more (esp. look-alike)
    BETA_TV     = 0.35
    GAMMA_FTL   = 1.33   # 1.5 if you need stronger focus

    LAM_2to1    = 0.40   # penalize look-alike → oil
    LAM_0to2    = 0.30   # penalize sea → look-alike
    GAMMA_CONF  = 2.0
    def loss_fn(logits, target):
        # 1) weighted CE (keeps calibration & early gradients healthy)
        ce = ce_loss(logits, target)

        # 2) Focal-Tversky (handles imbalance + hard examples)
        ftl = focal_tversky_loss(
            logits, target, alpha=ALPHA_TV, beta=BETA_TV, gamma=GAMMA_FTL
        )

        # 3) Targeted confusion shaping (do not let them dominate)
        pen_2to1 = confusion_penalty(
            logits, target, from_cls=2, to_cls=1, lam=LAM_2to1, gamma=GAMMA_CONF
        )
        pen_0to2 = confusion_penalty(
            logits, target, from_cls=0, to_cls=2, lam=LAM_0to2, gamma=GAMMA_CONF
        )
        return 0.3 * ce + 0.7 * ftl + pen_2to1 + pen_0to2
        # ---- Build a fixed probe batch for debug (deterministic) ----
    oil_spill_seg_model.eval()  # eval mode for consistent BN/Dropout in debug
    with torch.no_grad():
        probe_batch = next(iter(valid_dataset_loader))  # one small, fixed batch
    probe_images_dbg, probe_labels_dbg = probe_batch[0].to(device), probe_batch[1].to(device)
    print(f"[probe] images={tuple(probe_images_dbg.shape)}, labels={tuple(probe_labels_dbg.shape)}")
    oil_spill_seg_model.train()  # restore train mode before entering training loop
        # 4) Mix
    def _debug_loss_components(model, images_dbg, labels_dbg):
        model.eval()
        with torch.no_grad():
            logits_dbg = model(images_dbg)
            ce_dbg  = ce_loss(logits_dbg, labels_dbg).item()
            ftl_dbg = focal_tversky_loss(logits_dbg, labels_dbg,
                                        alpha=ALPHA_TV, beta=BETA_TV, gamma=GAMMA_FTL).item()
            p21_dbg = confusion_penalty(logits_dbg, labels_dbg, from_cls=2, to_cls=1,
                                        lam=LAM_2to1, gamma=GAMMA_CONF).item()
            p02_dbg = confusion_penalty(logits_dbg, labels_dbg, from_cls=0, to_cls=2,
                                        lam=LAM_0to2, gamma=GAMMA_CONF).item()
            total   = 0.3*ce_dbg + 0.7*ftl_dbg + p21_dbg + p02_dbg
            print(f"[dbg] CE={ce_dbg:.4f}  FTL={ftl_dbg:.4f}  2→1={p21_dbg:.4f}  0→2={p02_dbg:.4f}  total={total:.4f}")
        model.train()

    print("Warm-up debug on probe batch:")
    _debug_loss_components(oil_spill_seg_model, probe_images_dbg, probe_labels_dbg)
        
    print(f"\nTraining oil spill segmentation model: {FLAGS.which_model}\n")


    # Save training parameters locally and copy to Google Drive
    serializable_flags = {k: v for k, v in vars(FLAGS).items() if isinstance(v, (int, float, str, list, dict))}
    params_file_local = os.path.join(local_dir, "params.json")
    write_dict_to_json(params_file_local, serializable_flags)
    params_file_drive = os.path.join(dir_path, "params.json")
    shutil.copyfile(params_file_local, params_file_drive)

    early_stopping = EarlyStopping(patience=FLAGS.patience, verbose=True, path=os.path.join(dir_path, 'checkpoint.pt'))
    n_finetune_epochs = FLAGS.num_epochs  # how many epochs _more_ you want
    end_epoch   = start_epoch + n_finetune_epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        warm_frac = min(1.0, (epoch - start_epoch + 1) / max(1, int(0.4 * FLAGS.num_epochs)))  # first 40% epochs
        cur_scale = 0.1 + 0.9 * warm_frac  # start 0.1 → 1.0
        t_1 = time.time()
        train_loss = train_loop_mixed_ratio(
            train_loader_real, train_loader_synth,
            oil_spill_seg_model, loss_fn, optimizer, device,
            real_steps=getattr(FLAGS, "real_steps", 2),
            synth_steps=getattr(FLAGS, "synth_steps", 1),
            synth_loss_scale=cur_scale,   # your warm-up scaled synth weight
        )

        t_2 = time.time()
        print("-" * 100)
        print(
            f"Epoch : {epoch}/{FLAGS.num_epochs}, Time: {(t_2 - t_1):.2f} sec., Train Loss: {train_loss:.5f}"
        )
        valid_loss, valid_acc, valid_mIoU, valid_cls_IoU = validation_loop(
                valid_dataset_loader, oil_spill_seg_model, loss_fn, device,
                num_classes=FLAGS.num_classes,               # explicit & cheap
                ignore_index=getattr(FLAGS, "ignore_index", None)
            )

        if epoch == start_epoch or (epoch % 5 == 0):
            print(f"Probe debug @ epoch {epoch}:")
            _debug_loss_components(oil_spill_seg_model, probe_images_dbg, probe_labels_dbg)

        print(
            f"Validation Loss: {valid_loss:.5f}, "
            f"Pixel Acc: {valid_acc:.5f}, "
            f"mIoU: {valid_mIoU:.5f}"
        )
	        # Early stopping (if implemented)
        # early_stopping(valid_loss)
        early_stopping(valid_loss, oil_spill_seg_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Write metrics to CSV and flush immediately
        percls = [float(x) for x in valid_cls_IoU]
        if len(percls) < FLAGS.num_classes:
            percls += [float('nan')] * (FLAGS.num_classes - len(percls))
        elif len(percls) > FLAGS.num_classes:
            percls = percls[:FLAGS.num_classes]

        csv_writer.write_row(
            [
                int(epoch),
                round(float(train_loss.item()), 5),
                round(float(valid_loss.item()), 5),
                round(float(valid_acc), 5),
                round(float(valid_mIoU), 5),
                *[round(v, 5) for v in percls]  # c0..c4
            ]
        )
        # Flush and sync the CSV file
        csv_writer.file.flush()
        os.fsync(csv_writer.file.fileno())

        # Copy the metrics file to Google Drive
        metrics_file_local = os.path.join(local_dir, "train_metrics_ft.csv")

        shutil.copyfile(metrics_file_local, drive_metrics)

        # Wait until the metrics file is confirmed to be in Google Drive
        while not os.path.exists(drive_metrics ):
            print("Waiting for the metrics file to be copied to Google Drive...")
            time.sleep(1)
        print(f"Metrics file saved to {drive_metrics }")

        # Save model checkpoint locally

        local_model_checkpoint_path = os.path.join(local_dir, f"ft_{mode}_oil_spill_seg_{FLAGS.which_model}_{epoch}.pt")
        with open(local_model_checkpoint_path, 'wb') as f:
            torch.save(oil_spill_seg_model.state_dict(), f)
            f.flush()
            os.fsync(f.fileno())

        # Copy the model checkpoint to Google Drive
        drive_model_checkpoint_path = os.path.join(dir_path, f"ft_{mode}_oil_spill_seg_{FLAGS.which_model}_{epoch}.pt")
        shutil.copyfile(local_model_checkpoint_path, drive_model_checkpoint_path)
        
        
        state_path_local  = os.path.join(
            local_dir, f"ft_{mode}_oil_spill_seg_{FLAGS.which_model}_{epoch}.state")
        torch.save({
            'epoch': epoch,
            'model_state': oil_spill_seg_model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': lr_scheduler.state_dict()
        }, state_path_local)
        shutil.copyfile(state_path_local,
            os.path.join(dir_state,
                        f"ft_{mode}_oil_spill_seg_{FLAGS.which_model}_{epoch}.state"))
        _prune_states(dir_state, FLAGS.which_model,mode)  
        # 3) actualizar progreso global
        _update_progress(FLAGS.drive_destination_dir, FLAGS.which_model, epoch)


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
    shutil.copyfile(os.path.join(local_dir, "train_metrics_ft.csv"), os.path.join(dir_path, "train_metrics_ft.csv"))
    return