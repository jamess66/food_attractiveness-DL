import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import the model architecture
from model import SiameseNetwork

class FoodAttractivenessPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path (str): Path to the saved model checkpoint
            device (str, optional): Device to run prediction on
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = SiameseNetwork().to(self.device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path):
        """
        Load and transform an image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            torch.Tensor: Transformed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def predict(self, image1_path, image2_path):
        """
        Compare attractiveness of two images
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
        
        Returns:
            dict: Prediction results with probabilities and comparison
        """
        # Load images
        img1 = self.load_image(image1_path).to(self.device)
        img2 = self.load_image(image2_path).to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Get model outputs
            output1, output2 = self.model(img1, img2)
            
            # Convert to probabilities
            prob1 = output1[0, 1].cpu().numpy()
            prob2 = output2[0, 1].cpu().numpy()
        
        # Determine more attractive image
        if prob1 > prob2:
            winner = 'Image 1'
            confidence = prob1
            loser_prob = prob2
        else:
            winner = 'Image 2'
            confidence = prob2
            loser_prob = prob1
        
        return {
            'image1_attractiveness': prob1,
            'image2_attractiveness': prob2,
            'winner': winner,
            'winner_confidence': float(confidence),
            'loser_confidence': float(loser_prob)
        }

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Food Attractiveness Prediction')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--image1', type=str, required=True, 
                        help='Path to the first image')
    parser.add_argument('--image2', type=str, required=True, 
                        help='Path to the second image')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to run prediction (cuda/cpu)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = FoodAttractivenessPredictor(
            model_path=args.model, 
            device=args.device
        )
        
        # Predict
        result = predictor.predict(args.image1, args.image2)
        
        # Print results
        print("\n--- Food Attractiveness Prediction ---")
        print(f"Image 1 Attractiveness: {result['image1_attractiveness']:.4f}")
        print(f"Image 2 Attractiveness: {result['image2_attractiveness']:.4f}")
        print(f"\nWinner: {result['winner']}")
        print(f"Winner Confidence: {result['winner_confidence']:.4f}")
        print(f"Loser Confidence: {result['loser_confidence']:.4f}")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()


# python predict.py --model D:\cv_contest\food-attractive-DL\src\checkpoints\20250305_025201\best_model.pth --image1 D:\cv_contest\food-attractive-DL\dataset\images\b1_1.jpg --image2 D:\cv_contest\food-attractive-DL\dataset\images\b1_2.jpg