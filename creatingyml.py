# this code will create .yml file for the yolo model to train on it 
# dataset maker code 

import os
import shutil
import pandas as pd
import cv2
import yaml

# --- Configuration ---
IMAGES_DIR = 'images'
CSV_FILE = 'labels.csv'
OUTPUT_DIR = 'Pothole_Dataset'

# --- 1. Create Directories ---
train_img_path = os.path.join(OUTPUT_DIR, 'images', 'train')
train_lbl_path = os.path.join(OUTPUT_DIR, 'labels', 'train')
os.makedirs(train_img_path, exist_ok=True)
os.makedirs(train_lbl_path, exist_ok=True)

# --- 2. Process Data ---
df = pd.read_csv(CSV_FILE)
print(f"Processing {len(df['ImageID'].unique())} images...")

for image_id, group in df.groupby('ImageID'):
    source_image_path = os.path.join(IMAGES_DIR, image_id)
    if not os.path.exists(source_image_path):
        continue
    shutil.copy(source_image_path, os.path.join(train_img_path, image_id))

    image = cv2.imread(source_image_path)
    img_height, img_width, _ = image.shape

    yolo_annotations = []
    for _, row in group.iterrows():
        xmin, xmax, ymin, ymax = row['XMin'], row['XMax'], row['YMin'], row['YMax']
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_filename = os.path.splitext(image_id)[0] + '.txt'
    with open(os.path.join(train_lbl_path, label_filename), 'w') as f:
        f.write('\n'.join(yolo_annotations))

# --- 3. Create data.yaml File (THE FIX IS HERE) ---
yaml_data = {
    'train': os.path.abspath(train_img_path),
    'val': os.path.abspath(train_img_path),  # Point 'val' to the same 'train' path
    'nc': 1,
    'names': ['pothole']
}
with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"--- Dataset created successfully in '{OUTPUT_DIR}' folder. ---")
print("--- 'data.yaml' now includes a 'val' key as required. ---")