import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import active_contour
from noise import pnoise2

# Define the dataset class (adapted)
'''
class MaskOnlyDataset(Dataset):
    """
    Loads all *_label_1D.png in a directory as single-channel masks with integer {0..4}.
    """
    def __init__(self, label_dir, transform=None):
        self.label_dir = label_dir
        self.label_files = sorted([
            f for f in os.listdir(label_dir)
            if f.endswith('_label_1D.png')
        ])
        if not self.label_files:
            raise ValueError(f"No '_label_1D.png' files found in {label_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        filename = self.label_files[idx]
        full_path = os.path.join(self.label_dir, filename)
        mask_pil = Image.open(full_path).convert('L')
        mask_1d = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        if self.transform is not None:
            mask_1d = self.transform(mask_1d)
        return mask_1d
'''
class MaskOnlyDataset(Dataset):
    """
    Loads all *_mask1d_*.png in a directory as single-channel masks with integer {0..4}.
    """
    def __init__(self, label_dir, transform=None, strict_regex: bool = False):
        self.label_dir = label_dir

        if strict_regex:
            # e.g. 7_1_2015-03-16_S1_GRD_VV_mask1d_62.png
            pat = re.compile(r'.*_mask1d_\d+\.png$', re.IGNORECASE)
            files = [f for f in os.listdir(label_dir) if pat.match(f)]
        else:
            # looser: contains "_mask1d_" and ends with .png (case-insensitive)
            files = [f for f in os.listdir(label_dir)
                     if f.lower().endswith(".png") and "_mask1d_" in f.lower()]

        self.label_files = sorted(files)
        if not self.label_files:
            raise ValueError(f"No '*_mask1d_*.png' files found in {label_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        filename = self.label_files[idx]
        full_path = os.path.join(self.label_dir, filename)
        mask_pil = Image.open(full_path).convert('L')   # 8-bit single-channel
        mask_1d = torch.from_numpy(np.array(mask_pil, dtype=np.uint8)).to(torch.long)
        if self.transform is not None:
            mask_1d = self.transform(mask_1d)
        return mask_1d

# Cell 2: Define function to convert 1D mask to RGB using the label-to-RGB mapping

label_to_rgb = {
    0: [0, 0, 0],        # Sea Surface
    1: [0, 255, 255],    # Oil Spill
    2: [255, 0, 0],      # Look-alike
    3: [153, 76, 0],     # Ship
    4: [0, 153, 0]       # Land
}

def translate_1d_to_rgb(mask_1d):
    """
    Converts a 1D mask (H, W) with values {0,1,...,4} into an RGB image.
    """
    h, w = mask_1d.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in label_to_rgb.items():
        mask_rgb[mask_1d == label] = rgb
    return mask_rgb



# Cell 3: Function to extract connected regions for a given label

def extract_connected_regions(mask_1d,
                              label_value: int,
                              connectivity: int = 2,
                              min_area: int = 0,
                              top_k: int | None = None,
                              return_area: bool = True):
    """
    Extract connected regions for `label_value` and return them sorted by area (desc).

    Args
    ----
    mask_1d        : torch.Tensor[H,W] or np.ndarray[H,W]
    label_value    : class id to extract
    connectivity   : 1=4-connected, 2=8-connected (default)
    min_area       : drop regions with pixel count < min_area
    top_k          : keep only the largest k regions (None = all)
    return_area    : include 'area' in dicts

    Returns
    -------
    regions_info : List[dict] (largest first)
       dict keys: 'bbox', 'binary_mask', ('area' if return_area)
    """
    # accept torch or numpy
    if hasattr(mask_1d, "cpu"):
        label_mask = (mask_1d == label_value).cpu().numpy().astype(np.uint8)
    else:
        label_mask = (mask_1d == label_value).astype(np.uint8)

    labeled = cc_label(label_mask, connectivity=connectivity)
    regions_info = []

    for reg in regionprops(labeled):
        area = int(reg.area)
        if area < min_area:
            continue
        r1, c1, r2, c2 = reg.bbox
        region_mask_full = np.zeros_like(labeled, dtype=np.uint8)
        region_mask_full[reg.coords[:, 0], reg.coords[:, 1]] = 1
        entry = {
            "bbox": (r1, c1, r2, c2),
            "binary_mask": region_mask_full,
        }
        if return_area:
            entry["area"] = area
        regions_info.append(entry)

    # sort largest → smallest
    regions_info.sort(key=lambda d: d.get("area", 0), reverse=True)

    # keep only top_k if requested
    if top_k is not None:
        regions_info = regions_info[:top_k]

    return regions_info

def random_translate_region(mask_1d, label_value, max_shift=30, debug=False):
    """
    For a given label_value (e.g., 1), randomly translates each connected region.
    
    The function:
      1. Extracts connected regions.
      2. Computes each region's area and centroid.
      3. Sorts regions by descending area (largest moved first).
      4. For each region, computes a random translation vector (within max_shift radius).
      5. Moves the region's pixels accordingly (with clipping at the image boundary).
      6. Ensures that a new region does not overlap any previously placed (larger) region.
         If it would, that region is skipped.
    
    If debug=True, the function plots the original mask alongside the translated mask.
    
    Returns the new mask as a torch tensor.
    """
    # Convert mask to numpy (if not already) for easier manipulation
    original_np = mask_1d.cpu().numpy()
    new_mask = np.zeros_like(original_np)
    
    # Extract connected regions for the given label_value.
    regions_info = extract_connected_regions(mask_1d, label_value)
    
    # Gather details: area, centroid, and original coordinates.
    regions = []
    for region in regions_info:
        binary_mask = region['binary_mask']
        indices = np.argwhere(binary_mask == 1)  # pixel coordinates (row, col)
        area = indices.shape[0]
        centroid = indices.mean(axis=0)  # (row, col)
        regions.append({
            'bbox': region['bbox'],
            'binary_mask': binary_mask,
            'area': area,
            'centroid': centroid,
            'coords': indices
        })
    
    # Sort regions by descending area (largest first)
    regions_sorted = sorted(regions, key=lambda r: r['area'], reverse=True)
    
    H, W = original_np.shape
    # Process each region
    for reg in regions_sorted:
        # Compute a random shift vector:
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, max_shift)
        shift_y = int(round(radius * np.sin(angle)))
        shift_x = int(round(radius * np.cos(angle)))
        
        # New centroid (for reference, not used directly)
        new_centroid = reg['centroid'] + np.array([shift_y, shift_x])
        
        # Compute new coordinates for each pixel in the region
        new_coords = reg['coords'] + np.array([shift_y, shift_x])
        # Clip the coordinates so they lie within image boundaries
        new_coords[:, 0] = np.clip(new_coords[:, 0], 0, H - 1)
        new_coords[:, 1] = np.clip(new_coords[:, 1], 0, W - 1)
        
        # Check for overlap: ensure none of the new coordinates are already occupied
        conflict = False
        for coord in new_coords:
            if new_mask[coord[0], coord[1]] != 0:
                conflict = True
                break
        
        if not conflict:
            # Place region into new_mask at new coordinates
            for coord in new_coords:
                new_mask[coord[0], coord[1]] = label_value
        else:
            # Optionally, you can decide to skip or try a different shift.
            # Here, we simply skip if there's overlap.
            pass
    
    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_np, cmap='gray')
        plt.title("Original Mask")
        plt.subplot(1, 2, 2)
        plt.imshow(new_mask, cmap='gray')
        plt.title("Translated Mask")
        plt.tight_layout()
        plt.show()
    
    return torch.from_numpy(new_mask)
import random

def random_translate_and_rotate_region(mask_1d, label_value, max_shift=30, debug=False):
    """
    For a given label_value (e.g., 1), randomly translates and then rotates each connected region.
    
    Process:
      1. Extract connected regions for the given label.
      2. For each region (sorted by descending area), compute allowed translation ranges so that
         at least half of the region remains inside the image.
      3. Crop the region from its bounding box.
      4. Apply a random translation using cv2.warpAffine.
      5. Apply a random rotation (from 0, 90, 180, 270 degrees) using cv2.getRotationMatrix2D.
      6. Determine the new placement (top-left corner) in the full image.
      7. If the transformed region does not overlap any previously placed region, paste it into the new mask.
         (If there’s an overlap, that region is skipped.)
    
    Returns the new mask (as a torch tensor) with the augmented regions.
    """
    original_np = mask_1d.cpu().numpy()
    new_mask = np.zeros_like(original_np)
    
    # Extract connected regions for the label.
    regions_info = extract_connected_regions(mask_1d, label_value)
    
    # Build a list with region details.
    regions = []
    for region in regions_info:
        binary_mask = region['binary_mask']
        indices = np.argwhere(binary_mask == 1)
        area = indices.shape[0]
        bbox = region['bbox']  # (r1, c1, r2, c2)
        regions.append({'bbox': bbox, 'binary_mask': binary_mask, 'area': area})
    
    # Process larger regions first.
    regions_sorted = sorted(regions, key=lambda r: r['area'], reverse=True)
    H, W = original_np.shape
    
    for reg in regions_sorted:
        r1, c1, r2, c2 = reg['bbox']
        # Crop the region from its bounding box.
        region_crop = reg['binary_mask'][r1:r2, c1:c2]
        region_h, region_w = region_crop.shape
        
        # Compute allowed translation so that at least half the region remains.
        allowed_y_min = -min(max_shift, r1 + region_h // 2)
        allowed_y_max = min(max_shift, (H - r2) + region_h // 2)
        allowed_x_min = -min(max_shift, c1 + region_w // 2)
        allowed_x_max = min(max_shift, (W - c2) + region_w // 2)
        
        shift_y = int(round(np.random.uniform(allowed_y_min, allowed_y_max)))
        shift_x = int(round(np.random.uniform(allowed_x_min, allowed_x_max)))
        
        # Apply translation.
        M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated = cv2.warpAffine(region_crop, M_trans, (region_w, region_h), flags=cv2.INTER_NEAREST)
        
        # Apply a random rotation.
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            center = (region_w / 2, region_h / 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(translated, M_rot, (region_w, region_h), flags=cv2.INTER_NEAREST)
        else:
            rotated = translated.copy()
        
        # Determine the new top-left corner.
        new_r1 = r1 + shift_y
        new_c1 = c1 + shift_x
        new_r1 = np.clip(new_r1, 0, H - region_h)
        new_c1 = np.clip(new_c1, 0, W - region_w)
        new_r2 = new_r1 + region_h
        new_c2 = new_c1 + region_w
        
        # Check for overlap.
        region_coords = np.argwhere(rotated == 1)
        conflict = False
        for coord in region_coords:
            full_y = new_r1 + coord[0]
            full_x = new_c1 + coord[1]
            if new_mask[full_y, full_x] != 0:
                conflict = True
                break
        
        if not conflict:
            for coord in region_coords:
                full_y = new_r1 + coord[0]
                full_x = new_c1 + coord[1]
                new_mask[full_y, full_x] = label_value
        
        if debug:
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(region_crop, cmap='gray')
            plt.title("Original Region Crop")
            plt.subplot(1,3,2)
            plt.imshow(translated, cmap='gray')
            plt.title(f"Translated (shift_x={shift_x}, shift_y={shift_y})")
            plt.subplot(1,3,3)
            plt.imshow(rotated, cmap='gray')
            plt.title(f"Rotated ({angle}°)")
            plt.suptitle("Translation & Rotation Debug for a Region")
            plt.show()
    
    return torch.from_numpy(new_mask)

def random_translate_and_rotate_regionv2(region_mask, label_value, max_shift=30, debug=False):
    """
    Applies a random translation and then a random rotation to a given binary region mask.
    
    Inputs:
      - region_mask: a binary (0/1) numpy array representing the region.
      - label_value: label of the region (not used in transformation, but for clarity).
      - max_shift: maximum shift (the function computes allowed ranges based on image bounds).
      - debug: if True, displays intermediate outputs.
    
    Returns:
      - A torch tensor representing the transformed region (same shape as region_mask).
    """
    # Here we assume that region_mask is already cropped to its bounding box.
    region_h, region_w = region_mask.shape
    
    # For simplicity, we allow a full range of [-max_shift, max_shift] shifts.
    # (You can add extra logic here to compute allowed ranges.)
    shift_y = int(round(np.random.uniform(-max_shift, max_shift)))
    shift_x = int(round(np.random.uniform(-max_shift, max_shift)))
    
    M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(region_mask, M_trans, (region_w, region_h), flags=cv2.INTER_NEAREST)
    
    # Apply a random rotation from {0, 90, 180, 270}
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        center = (region_w / 2, region_h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(translated, M_rot, (region_w, region_h), flags=cv2.INTER_NEAREST)
    else:
        rotated = translated.copy()
    
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(region_mask, cmap='gray')
        plt.title("Original Cropped Region")
        plt.subplot(1,3,2)
        plt.imshow(translated, cmap='gray')
        plt.title(f"Translated (shift_x={shift_x}, shift_y={shift_y})")
        plt.subplot(1,3,3)
        plt.imshow(rotated, cmap='gray')
        plt.title(f"Rotated ({angle}°)")
        plt.suptitle(f"Label {label_value} Translation & Rotation")
        plt.show()
    
    return torch.from_numpy(rotated)