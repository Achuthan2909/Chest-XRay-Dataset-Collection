#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets

from MedViT import MedViT_small, MedViT_base, MedViT_large

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MedViT on your dataset')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'],
                        help='Size of MedViT model to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained weights (if available)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pre-trained weights')
    
    # Dataset parameters  
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing your dataset')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes in your dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (224 recommended for MedViT)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output models and logs')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def setup_model(args):
    # Initialize model based on size
    if args.model_size == 'small':
        model = MedViT_small(num_classes=args.num_classes)
    elif args.model_size == 'base':
        model = MedViT_base(num_classes=args.num_classes)
    elif args.model_size == 'large':
        model = MedViT_large(num_classes=args.num_classes)
    
    # Load pre-trained weights if specified
    if args.pretrained and args.pretrained_path:
        print(f"Loading pre-trained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Handle classifier mismatch: if shapes differ, randomly initialize classifier layer
        if model.proj_head[-1].weight.shape != state_dict['proj_head.0.weight'].shape:
            print(f"Classifier mismatch, initializing classifier layer randomly")
            del state_dict['proj_head.0.weight']
            del state_dict['proj_head.0.bias']
            
        model.load_state_dict(state_dict, strict=False)
    
    return model

def setup_data_loaders(args):
    
    print(f"Setting up data loaders...")
    print(f"Training data directory: {os.path.join(args.data_dir, 'train')}")
    print(f"Test data directory: {os.path.join(args.data_dir, 'test')}")
    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'test'),
        transform=test_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Found {len(train_dataset)} training samples")
    print(f"Found {len(test_dataset)} testing samples")
    
    return train_loader, test_loader

from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device): 
    model.train() 
    running_loss = 0.0 
    correct = 0 
    total = 0
    # Create a progress bar for the training loop
    pbar = tqdm(train_loader, desc='Training', leave=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calculate current average loss and accuracy
        avg_loss = running_loss / (total / labels.size(0))
        avg_acc = 100. * correct / total
        
        # Update the progress bar
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.2f}%'})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    print("Setting up random seeds...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    print("Creating output directory...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.model_size}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    print("Setting up device...")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Set up model
    print("Setting up model...")
    model = setup_model(args)
    model = model.to(device)
    
    ### Fine-tuning modifications:
    # Freeze the entire model
    print("Freezing model backbone and unfreezing classifier head and last few layers for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last 4 layers in the feature extractor. Adjust the slice ([-3:]) if needed.
    for layer in model.features[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Unfreeze the classifier head
    for param in model.proj_head.parameters():
        param.requires_grad = True
    
    # Set up data loaders
    print("Setting up data loaders...")
    train_loader, test_loader = setup_data_loaders(args)
    
    # Set up loss function and optimizer: Only parameters with requires_grad=True will be updated.
    print("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up learning rate scheduler
    print("Setting up learning rate scheduler...")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("Starting training...")
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate the model
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if test accuracy improved
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'args': args,
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'args': args,
            }
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Time: {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Log metrics to a file
        with open(os.path.join(output_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
    
    print(f"Training completed. Best test accuracy: {best_test_acc:.2f}%")
    print(f"Checkpoints saved to {output_dir}")

if __name__ == "__main__":
    main()