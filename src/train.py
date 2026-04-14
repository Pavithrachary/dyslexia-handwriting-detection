"""
train.py
--------
Training script for the dyslexia handwriting detection model.
"""

import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model


# ─── Configuration ────────────────────────────────────────────────────────────
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
MODEL_SAVE_PATH = 'results/dyslexia_handwriting_model.keras'
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
# ──────────────────────────────────────────────────────────────────────────────


def get_data_generators():
    """Create and return train and validation data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    )
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=CLASSES
    )
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, valid_generator


def train():
    os.makedirs('results', exist_ok=True)

    print("Loading data generators...")
    train_gen, valid_gen = get_data_generators()

    print("Building model...")
    model = build_model(num_classes=26)

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_SAVE_PATH}")
    return history


if __name__ == "__main__":
    train()
