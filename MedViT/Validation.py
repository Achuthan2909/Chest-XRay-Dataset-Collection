import os
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from MedViT import MedViT_small, MedViT_base, MedViT_large
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MedViT on validation dataset')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'],
                        help='Size of MedViT model to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if MPS is available')
    return parser.parse_args()

def get_device(force_cpu=False):
    if not force_cpu:
        if torch.backends.mps.is_available():
            try:
                mps_device = torch.device("mps")
                # Test MPS backend with a small tensor operation
                torch.ones(1).to(mps_device)
                print("Using MPS (Metal Performance Shaders) device")
                return mps_device
            except Exception as e:
                print(f"MPS is available but failed to initialize: {e}")
                print("Falling back to CPU")
    return torch.device("cpu")

def load_model(args, device):
    print(f"Loading MedViT-{args.model_size} model...")
    model = {'small': MedViT_small, 'base': MedViT_base, 'large': MedViT_large}[args.model_size](num_classes=args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model

def load_data(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                          num_workers=args.num_workers, pin_memory=True)
    return dataloader, dataset.classes

def evaluate(model, dataloader, device, class_names):
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Move results back to CPU for numpy conversion
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())
            
    return np.array(y_true), np.array(y_pred), np.array(y_scores), class_names

def plot_confusion_matrix(y_true, y_pred, class_names, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def main():
    args = parse_args()
    device = get_device(args.force_cpu)
    
    model = load_model(args, device)
    dataloader, class_names = load_data(args)
    y_true, y_pred, y_scores, class_names = evaluate(model, dataloader, device, class_names)
    
    # Print metrics
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # AUC-ROC Score
    if args.num_classes == 2:
        auc_score = roc_auc_score(y_true, y_scores[:, 1])
        print(f"AUC-ROC Score: {auc_score:.4f}")
    else:
        auc_score = roc_auc_score(y_true, y_scores, multi_class='ovr')
        print(f"AUC-ROC Score (multi-class): {auc_score:.4f}")
    
    # Save confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()