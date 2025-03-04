# predict.py
import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import FoodComparisonModel

class FoodAttractivenessPredictor:
    def __init__(self, model_path, backbone='convnext', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, backbone)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path, backbone):
        model = FoodComparisonModel(
            backbone_name=backbone,
            pretrained=False,  # Important: use False when loading trained weights
            freeze_backbone=False
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict_pair(self, image1_path, image2_path):
        with torch.no_grad():
            img1 = self.preprocess_image(image1_path)
            img2 = self.preprocess_image(image2_path)
            
            outputs = self.model(img1, img2)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            return {
                'image1_prob': probabilities[0].item(),
                'image2_prob': probabilities[1].item(),
                'prediction': 1 if probabilities[0] > probabilities[1] else 2
            }

def main():
    parser = argparse.ArgumentParser(description='Compare food attractiveness')
    parser.add_argument('--model', type=str, default="../cp/best_model_85.56-71.53.pth")
    parser.add_argument('--image1', type=str, required=True, help='First image path')
    parser.add_argument('--image2', type=str, required=True, help='Second image path')
    parser.add_argument('--backbone', type=str, default='convnext', 
                       help='Model backbone used during training')
    args = parser.parse_args()
    
    predictor = FoodAttractivenessPredictor(args.model, args.backbone)
    result = predictor.predict_pair(args.image1, args.image2)
    
    print(f"\nComparison Result:")
    print(f"Image 1 ({args.image1}): {result['image1_prob']*100:.1f}% confidence")
    print(f"Image 2 ({args.image2}): {result['image2_prob']*100:.1f}% confidence")
    print(f"\nConclusion: Image {result['prediction']} is more attractive!")

if __name__ == '__main__':
    main()


# python predict.py --model checkpoints/best_model.pth --image1 D:\cv_contest\food-attractive-DL\dataset\test\927497_231353200376295_1174550114_n.jpg --image2 D:\cv_contest\food-attractive-DL\dataset\test\927497_231353200376295_1174550114_n.jpg --backbone resnet50
