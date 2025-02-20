# Medical Chest X-Ray Dataset Analysis

This repository contains analysis and processing scripts for multiple chest X-ray datasets used for medical imaging research.

## Datasets Overview

### 1. RSNA Pneumonia Detection
- Source: RSNA Pneumonia Detection Challenge
- Classes: Normal, No Lung Opacity/Not Normal, Lung Opacity
- Data Format: DICOM images
- Additional Info: Includes bounding box annotations for lung opacities

### 2. VinBigData Chest X-Ray
- 15 Finding Categories including:
  - No Finding
  - Cardiomegaly
  - Aortic Enlargement
  - Pleural Thickening
  - And others
- Includes radiologist annotations (rad_id)
- Contains bounding box coordinates for abnormalities

### 3. Tuberculosis Datasets
#### a. Shenzhen Dataset
- Contains normal and tuberculosis cases
- Includes patient metadata:
  - Age
  - Gender
  - Findings
- Image Format: PNG

#### b. Montgomery Dataset
- Similar structure to Shenzhen
- Contains patient demographics and findings
- Higher resolution images (4020x4892)

## Repository Structure

### Analysis Notebooks
- `rsna_analysis.ipynb`: RSNA dataset analysis
- `vinbigdata_analysis.ipynb`: VinBigData chest X-ray analysis
- `tb_shenzhen_analysis.ipynb`: Shenzhen TB dataset analysis
- `tb_montgomery_analysis.ipynb`: Montgomery TB dataset analysis
- `nih_analysis.ipynb`: NIH dataset analysis
- `chexpert_analysis.ipynb`: CheXpert dataset analysis
