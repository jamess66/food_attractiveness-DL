import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FoodDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self):
        counts = np.bincount(self.df['winner'] - 1)  # Convert to 0-based index
        return torch.tensor([1.0 / c if c > 0 else 0 for c in counts], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = self._load_image(row['image_1'])
        img2 = self._load_image(row['image_2'])
        label = 0 if row['winner'] == 1 else 1
        return img1, img2, label

    def _load_image(self, filename):
        img = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        return self.transform(img) if self.transform else img

def create_dataloaders(config):
    df = pd.read_csv(os.path.join(config['data_root'], 'pairs.csv'))
    df['food_type'] = df['image_1'].str[0]
    
    # Stratified split by food type
    train_df, val_df = train_test_split(
        df,
        test_size=config['val_size'],
        stratify=df['food_type'],
        random_state=42
    )
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = FoodDataset(train_df, os.path.join(config['data_root'], 'images'), train_transform)
    val_dataset = FoodDataset(val_df, os.path.join(config['data_root'], 'images'), val_transform)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader