import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
import os
from dataset import create_datasets
from model import FoodComparisonModel

class FoodComparisonTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=2
        )
        self.train_loader, self.val_loader = self._prepare_dataloaders()
        
    def _build_model(self):
        model = FoodComparisonModel(
            backbone_name=self.config['backbone'],
            pretrained=self.config['pretrained'],
            freeze_backbone=self.config['freeze_backbone'],
            hidden_size=self.config['hidden_size'],
            dropout_prob=self.config['dropout_prob']
        )
        return model.to(self.device)
    
    def _prepare_dataloaders(self):
        train_dataset, val_dataset = create_datasets(self.config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(self.train_loader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f}')
                
        return total_loss/len(self.train_loader), 100*correct/total
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img1, img2, labels in self.val_loader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss/len(self.val_loader), 100*correct/total
    
    def train(self):
        best_acc = 0.0
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1:03d}')
            print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config['save_dir'], 'best_model.pth'))
                print('Saved new best model!')
                
            torch.save(self.model.state_dict(),
                      os.path.join(self.config['save_dir'], f'epoch_{epoch+1}.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    trainer = FoodComparisonTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()

# python train.py --config config.yaml