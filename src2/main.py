import torch
from torchvision import transforms
from PIL import Image


def compare_images(model, image1_path, image2_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img1 = transform(Image.open(image1_path)).unsqueeze(0).to(torch.device)
    img2 = transform(Image.open(image2_path)).unsqueeze(0).to(torch.device)
    
    with torch.no_grad():
        outputs = model(img1, img2)
        probabilities = torch.softmax(outputs, dim=1)
    
    if probabilities[0][0] > probabilities[0][1]:
        return f"Image 1 is more attractive ({probabilities[0][0]*100:.1f}%)"
    else:
        return f"Image 2 is more attractive ({probabilities[0][1]*100:.1f}%)"
    
import argparse

def main(args):
    model = torch.load(args.model)
    result = compare_images(model, args.image1, args.image2)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare attractiveness of two images')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--image1', type=str, required=True, help='Path to the first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second image')
    args = parser.parse_args()
    main(args)
