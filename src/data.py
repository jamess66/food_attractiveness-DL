import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_pairwise_data(csv_file, image_dir, img_size=(224, 224)):
    """ Load pairwise image dataset """
    df = pd.read_csv(csv_file)
    X, y = [], []
    
    for _, row in df.iterrows():
        img1 = cv2.imread(os.path.join(image_dir, row["Image 1"]))
        img2 = cv2.imread(os.path.join(image_dir, row["Image 2"]))
        
        if img1 is None or img2 is None:
            continue
        
        img1 = cv2.resize(img1, img_size) / 255.0
        img2 = cv2.resize(img2, img_size) / 255.0
        
        X.append(img1)
        X.append(img2)
        y.append(row["Winner"])
        y.append(1 - row["Winner"])  # Opposite label for the second image
    
    return np.array(X), np.array(y)