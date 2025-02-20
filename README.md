# Lung Disease Classification and Report Generation System

## Overview
This project develops an advanced deep learning-based system for detecting and classifying lung diseases using medical imaging data. The system combines state-of-the-art neural network architectures (UNET and ResNet) with visualization techniques (Grad-CAM, SHAP, and LIME) to provide accurate, interpretable results for clinical applications.

### Key Features
- Automated lung disease detection and classification
- Integration of multiple deep learning models (UNET, ResNet)
- Advanced visualization techniques (Grad-CAM, SHAP, LIME)
- Automated medical report generation
- Support for various medical imaging formats

## Prerequisites
- Python 3.13
- GPU with CUDA support (recommended)
- Access to dataset server (can access only if at achuthan's house)

## Installation

### 1. Clone the Repository
```bash
git clone [repository-url]
cd "Medical Report Generation"
```

### 2. Environment Setup

#### For Linux:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### For macOS:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Access

### Available Datasets (Open Access)

#### Chest X-ray Datasets
- [Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [VinBigData Chest X-ray Dataset](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
- [Montgomery County X-ray Set](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)
- [Shenzhen Hospital X-ray Set](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [CheXpert Dataset - Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

#### Report Generation Datasets
- [Curated CXR Report Generation Dataset](https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset)
- [Indiana PRO reports](https://www.kaggle.com/datasets/wasifnafee/indiana-pro-reports)

#### Other Datasets
- [ROCO Dataset (Radiology Objects in Context)](https://github.com/razorx89/roco-dataset)
- [OpenI-Yolov5-Bboxes](https://www.kaggle.com/datasets/mwnafee/openi-yolov5-bboxes)
   


### Remote Dataset Access

#### Linux/macOS Setup
To mount the remote dataset:
```bash
sudo mount -o vers=4,resvport,rw 192.168.0.193:/mnt/storage/Datasets "/path/to/local/mount/point"
```

Note: Replace "/path/to/local/mount/point" with your desired local directory path.

#### Windows Setup
1. Enable NFS Client:
   - Open Control Panel
   - Go to Programs and Features
   - Click "Turn Windows features on or off"
   - Enable "Services for NFS"
   - Restart your system

2. Mount the drive using PowerShell (Run as Administrator):
```powershell
mount -o anon 192.168.0.193:/mnt/storage/Datasets Z:
```

### Network Requirements
- The remote dataset is only accessible within the local network (192.168.0.x)
- Ensure proper network connectivity and permissions before attempting to mount
- Contact achuthan for access credentials if required
- Check if you can SSH in to the server first or try pinging the server

## Project Structure
```
Medical Report Generation/
├── models/
│   ├── unet/
│   └── resnet/
├── utils/
│   ├── visualization/
│   └── preprocessing/
├── remote-dataset/ (You have to create this then mount it inside this directory)
├── configs/
└── dataset-info/
    └── notebooks/   
```

## Contributors
- Sarvesh R K (21BCE5732)
- Achuthan R (21BAI1449)
- Varun R (21BAI1577)
- Guide: Prof. Kadhar Nawas K
