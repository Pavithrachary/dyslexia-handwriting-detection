"""
model.py
--------
Defines the ResNet50-based model architecture for dyslexia handwriting classification.
"""

from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


def build_model(num_classes: int = 26, input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Builds a transfer learning model using ResNet50 as the backbone.

    Args:
        num_classes (int): Number of output classes. Default is 26 (A-Z).
        input_shape (tuple): Input image dimensions. Default is (224, 224, 3).

    Returns:
        model (keras.Model): Compiled Keras model.
    """
    # Load pre-trained ResNet50 (frozen base)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
