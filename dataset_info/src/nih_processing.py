import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Define paths
nih_base_path = "/Users/anyhow/projects/data_science/Medical-Report-Generation/remote-dataset/NIH-Chest-X-rays"
image_folders = [f"images_{str(i).zfill(3)}" for i in range(1, 13)]  # images_001 to images_012
metadata_path = os.path.join(nih_base_path, "Data_Entry_2017.csv")

# Output paths
output_base_path = '/Users/anyhow/projects/data_science/Medical-Report-Generation/processed_dataset'
output_train_path = os.path.join(output_base_path, "train")
os.makedirs(output_train_path, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)

# Extract image-to-label mapping
image_label_mapping = dict(zip(df["Image Index"], df["Finding Labels"]))

# Get unique labels, ensure proper trimming of whitespace
all_labels = []
for labels_str in df["Finding Labels"].unique():
    for label in labels_str.split("|"):
        all_labels.append(label.strip())  # Strip whitespace
unique_labels = set(all_labels)

# Create output folders for each label
label_to_folder = {}  # Keep track of normalized label names
for label in unique_labels:
    normalized_label = label.strip().replace("/", "_").replace(" ", "_").lower()
    label_folder = os.path.join(output_train_path, normalized_label)
    os.makedirs(label_folder, exist_ok=True)
    label_to_folder[label.strip()] = normalized_label  # Store mapping

def process_image(image_name):
    try:
        # Find the image file in the dataset
        for folder in image_folders:
            image_path = os.path.join(nih_base_path, folder, image_name)
            if os.path.exists(image_path):
                break
        else:
            print(f"Image not found: {image_name}")
            return None
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_name}")
            return None
        
        # Resize image to 224x224
        img_resized = cv2.resize(img, (224, 224))
        
        # Normalize pixel values
        img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min()) * 255.0
        img_resized = img_resized.astype(np.uint8)
        
        # Get labels for the image and strip whitespace
        labels = [label.strip() for label in image_label_mapping[image_name].split("|")]
        
        # Save image in each corresponding label folder
        for label in labels:
            normalized_label = label_to_folder[label]  # Use the mapping
            output_file = os.path.join(output_train_path, normalized_label, image_name)
            cv2.imwrite(output_file, img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        return image_name
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        return None

# Process images in parallel
num_workers = os.cpu_count()
print(f"Using {num_workers} workers for parallel processing")

image_list = list(image_label_mapping.keys())
processed_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_image = {executor.submit(process_image, img_name): img_name for img_name in image_list}
    
    for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_list)):
        image_name = future_to_image[future]
        try:
            result = future.result()
            if result:
                processed_count += 1
        except Exception as exc:
            print(f'{image_name} generated an exception: {exc}')

print(f"Successfully processed {processed_count} images")
print("Processing completed!")