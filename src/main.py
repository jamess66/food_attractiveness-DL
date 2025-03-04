import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FoodAttractivenessModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FoodAttractivenessModel, self).__init__()
        # Use a pre-trained ResNet50 as the base feature extractor
        base_model = models.resnet50(pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Custom classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Classify attractiveness
        attractiveness = self.classifier(features)
        return attractiveness

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

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.food_model = FoodAttractivenessModel()
    
    def forward(self, image1, image2):
        # Get attractiveness scores for both images
        output1 = self.food_model(image1)
        output2 = self.food_model(image2)
        
        return output1, output2

def custom_loss(output1, output2, labels):
    """
    Custom loss function to compare attractiveness
    Labels: 0 means second image is more attractive
            1 means first image is more attractive
    """
    # Compute difference in attractiveness scores
    diff = torch.abs(output1[:, 1] - output2[:, 1])
    
    # Compute ranking loss
    ranking_loss = torch.mean(
        torch.where(
            labels == 1, 
            torch.relu(0.5 - diff),   # Penalize if first image is not clearly more attractive
            torch.relu(diff - 0.5)    # Penalize if second image is not clearly more attractive
        )
    )
    
    return ranking_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50):
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            
            # Compute loss
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                output1, output2 = model(img1, img2)
                
                # Compute loss
                loss = criterion(output1, output2, labels)
                val_loss += loss.item()
                
                # Compute accuracy
                # If first image is labeled as more attractive
                pred = (output1[:, 1] > output2[:, 1]).long()
                correct_predictions += (pred == labels).sum().item()
                total_predictions += labels.size(0)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {correct_predictions/total_predictions:.4f}')

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.0001
    epochs = 50
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    csv_path = 'path/to/your/dataset.csv'
    image_dir = 'path/to/your/image/directory'
    
    # Create datasets
    full_dataset = FoodImageComparisonDataset(csv_path, image_dir)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SiameseNetwork()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, custom_loss, optimizer, device, epochs)
    
    # Save the model
    torch.save(model.state_dict(), 'food_attractiveness_model.pth')
    
    # Prediction function (example)
    def predict_attractiveness(image1, image2):
        model.eval()
        with torch.no_grad():
            output1, output2 = model(image1.unsqueeze(0), image2.unsqueeze(0))
            # Return probabilities of attractiveness for each image
            return output1[0, 1].item(), output2[0, 1].item()

if __name__ == '__main__':
    main()