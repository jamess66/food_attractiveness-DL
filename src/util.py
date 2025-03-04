import torch

# Assuming your model is imported from model.py
from model import ResNetCustom, SiameseNetwork

def print_model_summary(model):
    """
    Print detailed model summary using print statements
    """
    print("Model Structure:")
    print(model)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

def main():
    # Create model instance
    model = SiameseNetwork()
    
    # Method 1: Print basic model structure
    print_model_summary(model)

if __name__ == '__main__':
    main()