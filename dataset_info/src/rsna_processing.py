import os
import pandas as pd
import pydicom
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading

# Thread local storage for pydicom cache
thread_local = threading.local()

# Define paths
train_images_path = "/Users/anyhow/projects/data_science/Medical-Report-Generation/remote-dataset/rsna-pneumonia-detection-challenge/stage_2_train_images"
detailed_class_labels_path = "/Users/anyhow/projects/data_science/Medical-Report-Generation/remote-dataset/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"  # This should be the file with patientId,class

# Create output directories
output_base_path = "/Users/anyhow/projects/data_science/Medical-Report-Generation/processed_dataset"
output_train_path = os.path.join(output_base_path, "train")

# Ensure output directories exist
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(output_train_path, exist_ok=True)

# Load labels
detailed_class_df = pd.read_csv(detailed_class_labels_path)

# Create a mapping of patientId to detailed class
class_mapping = detailed_class_df.set_index('patientId')['class'].to_dict()

# Create class folders upfront
unique_classes = detailed_class_df['class'].unique()
for class_name in unique_classes:
    class_folder = class_name.replace('/', '_').replace(' ', '_').lower()
    os.makedirs(os.path.join(output_train_path, class_folder), exist_ok=True)

# Function to read and preprocess DICOM image
def preprocess_dicom(file_path, target_size=(224, 224)):
    # Read DICOM file
    dicom = pydicom.dcmread(file_path)
    img = dicom.pixel_array
    
    # Convert to grayscale if needed (likely already grayscale)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Normalize pixel values
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    
    # Resize to target size
    img_resized = cv2.resize(img, target_size)
    
    return img_resized

# Worker function for training images
def process_train_image(patient_id):
    try:
        # Skip if patient doesn't have class info
        if patient_id not in class_mapping:
            print(f"Warning: No class information found for patient {patient_id}")
            return None
        
        detailed_class = class_mapping[patient_id]
        class_folder = detailed_class.replace('/', '_').replace(' ', '_').lower()
        
        dicom_path = os.path.join(train_images_path, f"{patient_id}.dcm")
        if not os.path.exists(dicom_path):
            print(f"DICOM file not found for patient {patient_id}")
            return None
        
        img_resized = preprocess_dicom(dicom_path)
        output_file = os.path.join(output_train_path, class_folder, f"{patient_id}.png")
        cv2.imwrite(output_file, img_resized)
        return patient_id
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return None

# Get the number of workers (typically CPU cores)
num_workers = os.cpu_count()
print(f"Using {num_workers} workers for parallel processing")

# Process training images in parallel
print("Processing training images...")
patient_ids = list(class_mapping.keys())
processed_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_patient = {executor.submit(process_train_image, patient_id): patient_id for patient_id in patient_ids}
    
    for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patient_ids)):
        patient_id = future_to_patient[future]
        try:
            result = future.result()
            if result:
                processed_count += 1
        except Exception as exc:
            print(f'{patient_id} generated an exception: {exc}')

print(f"Successfully processed {processed_count} training images")
print("Processing completed!")