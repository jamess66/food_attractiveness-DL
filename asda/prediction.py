import cv2
import numpy as np


def predict_attractiveness(model, img1_path, img2_path):
    def preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img.astype(np.float32) / 255.0
    
    img1 = preprocess(img1_path)[np.newaxis, ...]  # Add batch dimension
    img2 = preprocess(img2_path)[np.newaxis, ...]
    
    prob = model.predict([img1, img2])[0][0]
    
    if prob > 0.5:
        return f"Image 1 is more attractive ({prob*100:.1f}% confidence)"
    else:
        return f"Image 2 is more attractive ({(1-prob)*100:.1f}% confidence)"