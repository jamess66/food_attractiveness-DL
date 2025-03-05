from tensorflow.keras import applications, layers, models, Input, Model

def create_siamese_model():
    # Base network
    base_cnn = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze initial layers
    for layer in base_cnn.layers[:100]:
        layer.trainable = False
        
    # Feature extraction
    input_a = Input(shape=(224, 224, 3))
    input_b = Input(shape=(224, 224, 3))
    
    processed_a = base_cnn(input_a)
    processed_b = base_cnn(input_b)
    
    # Attention mechanism
    gap_a = layers.GlobalAveragePooling2D()(processed_a)
    gap_b = layers.GlobalAveragePooling2D()(processed_b)
    
    # Feature difference
    diff = layers.Subtract()([gap_a, gap_b])
    
    # Classification head
    x = layers.Dense(256, activation='relu')(diff)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=[input_a, input_b], outputs=outputs)

model = create_siamese_model()