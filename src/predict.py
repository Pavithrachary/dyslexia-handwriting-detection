"""
predict.py
----------
Run inference on a single image or a directory of images.

Usage:
    python src/predict.py --image path/to/image.png
"""

import argparse
import numpy as np
from PIL import Image
from keras.models import load_model


MODEL_PATH = 'results/dyslexia_handwriting_model.keras'
CLASS_NAMES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    img = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


def predict(image_path: str) -> str:
    """Returns the predicted character class for an image."""
    model = load_model(MODEL_PATH)
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_index] * 100
    label = CLASS_NAMES[predicted_index]
    print(f"Predicted Letter : {label}")
    print(f"Confidence       : {confidence:.2f}%")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten character from image.")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    predict(args.image)
