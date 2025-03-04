import os
import yaml
import torch
from argparse import ArgumentParser
from models.food_model import FoodComparisonModel
from utils.dataset import create_dataloaders
from utils.trainer import FoodTrainer

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = FoodComparisonModel(
        backbone_name=args.backbone,
        pretrained=True,
        emb_dim=config['model']['emb_dim'],
        dropout=config['model']['dropout'],
        num_attention_heads=config['model']['num_attention_heads']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config['data'])
    
    # Initialize trainer
    trainer = FoodTrainer(model, train_loader, val_loader, config['training'])
    
    # Training loop
    os.makedirs(config['save_dir'], exist_ok=True)
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc = trainer.validate()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save best model
        if val_acc > trainer.best_acc:
            trainer.best_acc = val_acc
            trainer.save_checkpoint(os.path.join(config['save_dir'], 'best_model.pth'))
        
        # Save periodic checkpoint
        if (epoch+1) % 10 == 0:
            trainer.save_checkpoint(os.path.join(config['save_dir'], f'epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    main()