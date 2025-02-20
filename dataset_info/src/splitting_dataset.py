import os
import shutil
from sklearn.model_selection import train_test_split
import random

def split_dataset(source_dir, train_size=0.5, val_size=0.3, test_size=0.2, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        source_dir (str): Path to the source directory containing disease folders
        train_size (float): Proportion of data for training
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    """
    # Create main output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(source_dir, split), exist_ok=True)
    
    # Process each disease folder
    disease_folders = [f for f in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, f)) 
                      and f not in splits]
    
    for disease in disease_folders:
        disease_path = os.path.join(source_dir, disease)
        files = os.listdir(disease_path)
        
        # First split: separate train and temp (val + test)
        train_files, temp_files = train_test_split(
            files, 
            train_size=train_size,
            random_state=random_state
        )
        
        # Second split: separate val and test from temp
        relative_val_size = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=relative_val_size,
            random_state=random_state
        )
        
        # Create disease folders in each split directory
        for split in splits:
            os.makedirs(os.path.join(source_dir, split, disease), exist_ok=True)
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy2(
                os.path.join(disease_path, file),
                os.path.join(source_dir, 'train', disease, file)
            )
        
        for file in val_files:
            shutil.copy2(
                os.path.join(disease_path, file),
                os.path.join(source_dir, 'val', disease, file)
            )
            
        for file in test_files:
            shutil.copy2(
                os.path.join(disease_path, file),
                os.path.join(source_dir, 'test', disease, file)
            )

if __name__ == "__main__":
    source_directory = "/Users/anyhow/projects/data_science/Medical-Report-Generation/processed_dataset"
    split_dataset(source_directory)
    print("Dataset splitting completed!")