<<<<<<< HEAD
# MorpSynth
=======
# 1. MORP–Synth Pipeline Overview

**Goal:** Improve cross-domain generalization (**Mediterranean → Peru**) by combining:

- **Geometric mask perturbations (MORP)**
- **SAR-aware synthetic generation (INADE)**  
- **Mixed real + synthetic segmentation training**

---

## Pipeline Summary

1. **Start** with a real Sentinel-1 SAR patch and its semantic mask.

2. **Stage A — MORP**  
   Applies controlled geometric perturbations to oil & look-alike regions:
   - rotation & translation  
   - apex detection  
   - curvature-driven bulges/shrinks

3. **Stage B — INADE**  
   Generates SAR-like textures from the edited masks while preserving spatial semantics.

4. **Stage C — Segmentation**  
   Train segmentation models using a mixture of:
   - real SAR patches  
   - real masks  
   - synthetic SAR + synthetic masks (from MORP + INADE)

**Result:** This mixed strategy significantly improves robustness to **domain shift**.


## 🔹 Step 1 — Train the INADE cGAN (Label → SAR)

Train INADE on the real dataset to learn **mask-to-SAR** generation. See original repository: https://github.com/tzt101/INADE 

```bash
python INADE/train.py --name derrame_exp_INADE512 \
  --dataset_mode oilspill \
  --norm_mode inade \
  --use_vae \
  --use_amp \
  --z_dim 512 \
  --eval_epoch_freq 5 \
  --dataroot "" \
  --gpu_ids -1 \
  --batchSize 1 \
  --niter 15 \
  --niter_decay 30 \
  --load_size 512 \
  --crop_size 512 \
  --label_nc 5
```

## 🔹 Step 2 — Apply MORP Mask Transformations

Generate geometric mask augmentations for:

Run morp_transformations/morp_code_spikeaware_v4.py for augmenting labels {1,2} of domain dataset.
Output:
MORP-augmented masks (e.g., morp_masks/patch001_aug3.png)

These masks are the inputs for INADE synthesis. Apply the process with INADE\test.py
```bash
python test.py --name derrame_exp_INADEPERUANO512_corrected_v2 --dataset_mode oilspillperu2synth --norm_mode inade --use_vae --z_dim 512 --dataroot "" --results_dir "" --which_epoch latest --how_many 902 --gpu_ids -1 --load_size 512 --crop_size 512 --batchSize 2
```

## 🔹 Step 3 — Train the Segmentation Model (Real + Synthetic)

Run segmentation/train_hpc_finetune_peru_v3_synth_v3_v1.py with the appropiate dirs for data location.

>>>>>>> master
