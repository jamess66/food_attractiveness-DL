import torch
from tqdm import tqdm

class FoodTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.train_loader.dataset.class_weights.to(self.device),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': config['lr']/10},
            {'params': model.attention.parameters(), 'lr': config['lr']},
            {'params': model.comparison.parameters(), 'lr': config['lr']}
        ], weight_decay=config.get('weight_decay', 1e-4))
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['epochs']
        )
        
        self.best_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader), correct / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for img1, img2, labels in self.val_loader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                outputs = self.model(img1, img2)
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(self.val_loader), correct / len(self.val_loader.dataset)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc
        }, path)