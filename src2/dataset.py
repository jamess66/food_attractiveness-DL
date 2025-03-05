import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class FoodPairDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pair = self.df.iloc[idx]
        img1_path = os.path.join(self.root_dir, pair['image_1'])
        img2_path = os.path.join(self.root_dir, pair['image_2'])
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        label = 0 if pair['winner'] == 1 else 1
        return img1, img2, label

def create_datasets(config):
    df = pd.read_csv(os.path.join(config['data_root'], 'winner.csv'))
    
    df['food_type'] = df['image_1'].str[0]
    
    train_df, val_df = train_test_split(
        df,
        test_size=config['val_size'],
        stratify=df['food_type'],
        random_state=42
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,  # Subtle brightness changes
            contrast=0.2,    # Subtle contrast variations
            saturation=0.2,  # Preserve food color vibrancy
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = FoodPairDataset(
        train_df,
        os.path.join(config['data_root'], 'images'),
        train_transform
    )
    
    val_dataset = FoodPairDataset(
        val_df,
        os.path.join(config['data_root'], 'images'),
        val_transform
    )
    
    return train_dataset, val_dataset


def main():
    # Test dataset splitting
    import yaml
    
    # Load configuration from YAML file
    with open('src2/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    train_dataset, val_dataset = create_datasets(config)
    print("Train dataset filenames:", train_dataset.df['image_1'].tolist() + train_dataset.df['image_2'].tolist())
    print("Validation dataset filenames:", val_dataset.df['image_1'].tolist() + val_dataset.df['image_2'].tolist())

if __name__ == '__main__':
    main()
