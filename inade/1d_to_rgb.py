import os
from PIL import Image
import numpy as np

# Your color mapping
dict_label_to_color_mapping = {
    0: np.array([0, 0, 0]),
    1: np.array([0, 255, 255]),
    2: np.array([255, 0, 0]),
    3: np.array([153, 76, 0]),
    4: np.array([0, 153, 0]),
}

# Paths
src_folder = r"D:\Derrame_Data\peru_split_8020__corregidas_v3_stride384_INADE\train\labels1d"
dst_folder = os.path.join(os.path.dirname(src_folder), "labels")
os.makedirs(dst_folder, exist_ok=True)

# Convert all .png files
for fname in os.listdir(src_folder):
    if fname.endswith(".png"):
        label_path = os.path.join(src_folder, fname)
        label_img = Image.open(label_path).convert("L")
        label_np = np.array(label_img)

        rgb_img = np.zeros((*label_np.shape, 3), dtype=np.uint8)
        for class_id, color in dict_label_to_color_mapping.items():
            rgb_img[label_np == class_id] = color

        Image.fromarray(rgb_img).save(os.path.join(dst_folder, fname))

print("✅ RGB masks saved in:", dst_folder)
