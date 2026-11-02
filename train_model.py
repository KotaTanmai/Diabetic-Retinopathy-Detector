
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# Load CSV files
train_df = pd.read_csv("dataset/train_split.csv")
val_df = pd.read_csv("dataset/val_split.csv")

# Function to load and preprocess images
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0

    return image, tf.cast(label, tf.float32)

# Create TensorFlow datasets
def create_dataset(data, training=False):
    paths = data["image_path"].values
    labels = data["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(len(data))

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = create_dataset(train_df, True)
val_ds = create_dataset(val_df)

# Handle class imbalance
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_df["label"].values
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# Save the best model
os.makedirs("model", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/retinopathy_model.keras",
    save_best_only=True,
    monitor="val_loss"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train CNN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

print("\nCNN training completed!")
