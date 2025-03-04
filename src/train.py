import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import logging
from datetime import datetime

# Import custom model (assuming it's in the same directory)
from model import ResNetCustom, SiameseNetwork, custom_loss

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

class FoodImageComparisonDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image pairs
            image_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load images
        img1_name = self.data.iloc[idx]['Image 1']
        img2_name = self.data.iloc[idx]['Image 2']
        winner = self.data.iloc[idx]['Winner']
        
        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)
        
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)
        
        return img1, img2, torch.tensor(winner, dtype=torch.long)
    
    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for img1, img2, labels in train_loader:
        # Move data to device
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output1, output2 = model(img1, img2)
        
        # Compute loss
        loss = criterion(output1, output2, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        pred = (output1[:, 1] > output2[:, 1]).long()
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
    
    # Compute average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """
    Validate model performance
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            # Move data to device
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            
            # Compute loss
            loss = criterion(output1, output2, labels)
            
            # Compute accuracy
            pred = (output1[:, 1] > output2[:, 1]).long()
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
    
    # Compute average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def main(args):
    """
    Main training script
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join('checkpoints', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset and DataLoaders
    full_dataset = FoodImageComparisonDataset(
        csv_file=args.dataset, 
        image_dir=args.image_dir
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = SiameseNetwork().to(device)
    
    # Loss and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    logging.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        # Log results
        logging.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_model_path)
            logging.info(f"Saved new best model to {best_model_path}")
    
    logging.info("Training completed!")

def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Food Attractiveness Comparison Model Training')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to CSV file with image pairs')
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Directory containing images')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay (L2 penalty)')
    
    # Device and computation
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA training')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


# python train.py --dataset D:\cv_contest\food-attractive-DL\dataset\dataset.csv --image_dir D:\cv_contest\food-attractive-DL\dataset\images --epochs 5 --batch_size 32 --learning_rate 0.0001
