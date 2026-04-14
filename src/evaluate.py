"""
evaluate.py
-----------
Evaluates the trained model on the test set and produces metrics + plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


TEST_DIR = 'dataset/test'
MODEL_PATH = 'results/dyslexia_handwriting_model.keras'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_test_generator():
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.show()
    print("Saved: results/training_curves.png")


def evaluate(history=None):
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading test data...")
    test_gen = load_test_generator()

    print("Evaluating...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    Y_pred = model.predict(test_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print("\n─── Classification Report ───")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    if history:
        plot_training_history(history)


if __name__ == "__main__":
    evaluate()
