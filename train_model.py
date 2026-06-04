import tensorflow as tf
import os

# --- TWEAK YOUR TRAINING HYPERPARAMETERS HERE ---
AUGMENTED_DATA_DIR = os.path.join("dataset", "augmented")
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 6
# Lowered learning rate for precise, stable fine-tuning adjustments
LEARNING_RATE = 0.0001  

print("Loading pre-augmented dataset into memory...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    AUGMENTED_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    AUGMENTED_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# --- BUILD THE NETWORKING BRAIN (WITH FINE-TUNING) ---
print("\nInitializing MobileNetV2 base architecture...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)

# 1. Unfreeze the base model so we can alter structural parameters
base_model.trainable = True

# 2. Freeze all early layers EXCEPT the top 20 layers
# This protects Google's edge-detection logic while letting us tune high-level feature maps
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Construct final network layers
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)), # Fast pixel normalization
    base_model,                                                   # Unfrozen MobileNetV2 Core
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),                                 # Prevents training overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')                # 0.0 to 1.0 confidence output
])

# Configure optimizer and loss functions with our safer learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\nTraining model on the augmented dataset for {EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# --- CONVERT TO EXPORTABLE TFLITE ---
print("\nConverting optimized model architecture to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nFinished! 'model.tflite' has been generated and is ready for live testing.")