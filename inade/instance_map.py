import numpy as np
from scipy.ndimage import label as connected_components
from PIL import Image
import os

def build_instance_map(label_np):
    instance_map = np.zeros_like(label_np, dtype=np.uint16)
    current_instance_id = 1

    # Only process classes 1, 2, 3
    for class_id in [1, 2, 3]:
        if class_id not in np.unique(label_np):
            continue
        mask = (label_np == class_id).astype(np.uint8)
        labeled_array, num_features = connected_components(mask)
        if num_features > 0:
            labeled_array[labeled_array > 0] += current_instance_id
            instance_map += labeled_array.astype(np.uint16)
            current_instance_id += num_features
    return instance_map

# Example usage
input_folder = r"D:\Derrame_Data\selected_sampling_final_less_bg_addmoresea_trainINADE\train\labels_1D"  # semantic masks (0 to 4)
output_folder = r"D:\Derrame_Data\selected_sampling_final_less_bg_addmoresea_trainINADE\train\inst"    # where to save the instance maps

os.makedirs(output_folder, exist_ok=True)

for fname in sorted(os.listdir(input_folder)):
    if not fname.endswith('.png'):
        continue
    label = Image.open(os.path.join(input_folder, fname)).convert('L')
    label_np = np.array(label)
    
    inst_map = build_instance_map(label_np)

    # Save as 16-bit PNG to preserve IDs > 255
    inst_img = Image.fromarray(inst_map.astype(np.uint16))
    inst_img.save(os.path.join(output_folder, fname))

print("✅ Instance maps generated correctly.")