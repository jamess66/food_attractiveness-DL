# Create data generators
import pandas as pd
import tensorflow as tf
from model import create_siamese_model
from dataset import FoodPairGenerator
import dataset
from sklearn.model_selection import train_test_split

# Load main dataframe
full_df = pd.read_csv('D:/food_attractiveness-DL/dataset/winner.csv')

# Split into train/validation
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

# Create generators
train_gen = FoodPairGenerator(
    dataframe=train_df,
    image_dir='D:/food_attractiveness-DL/dataset/images',
    batch_size=32,
    img_size=(224, 224)
)

val_gen = FoodPairGenerator(
    dataframe=val_df,
    image_dir='dataset/images',
    batch_size=32,
    img_size=(224, 224),
    shuffle=False
)

model = create_siamese_model()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)