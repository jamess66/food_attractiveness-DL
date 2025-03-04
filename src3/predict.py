import torch
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
from models.food_model import FoodComparisonModel

def predict():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image1', type=str, required=True)
    parser.add_argument('--image2', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    args = parser.parse_args()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FoodComparisonModel(backbone_name=args.backbone, pretrained=False)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process images
    img1 = transform(Image.open(args.image1).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(args.image2).convert('RGB')).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img1, img2)
        probs = torch.softmax(output, dim=1)
    
    print(f"\nImage 1 ({args.image1}): {probs[0][0].item():.2%}")
    print(f"Image 2 ({args.image2}): {probs[0][1].item():.2%}")
    print(f"\nPredicted winner: Image {1 if probs[0][0] > probs[0][1] else 2}")

if __name__ == '__main__':
    predict()