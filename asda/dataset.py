import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class FoodPairGenerator(Sequence):
    def __init__(self, dataframe, image_dir, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch = self.df.iloc[batch_indices]
        
        image1_batch = []
        image2_batch = []
        labels = []
        
        for _, row in batch.iterrows():
            img1 = self.load_image(row['image_1'])
            img2 = self.load_image(row['image_2'])
            label = 1 if row['winner'] == 1 else 0  # Convert to binary label
            
            image1_batch.append(img1)
            image2_batch.append(img2)
            labels.append(label)
            
        return [np.array(image1_batch), np.array(image2_batch)], np.array(labels)
    
    def load_image(self, filename):
        img = cv2.imread(str(self.image_dir / filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0  # Normalize
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)