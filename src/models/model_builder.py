"""
For creating CNN model
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import Config


class ModelBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build_base_model(self) -> tf.keras.Model:
        """Load pre-trained base model (ResNet50 or MobileNetV2)"""
        if self.config.BASE_MODEL == 'resnet50':
            base_model = keras.applications.ResNet50(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.config.BASE_MODEL == 'mobilenetv2':
            base_model = keras.applications.MobileNetV2(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {self.config.BASE_MODEL}")

        # Freeze base model layers for transfer learning
        base_model.trainable = False
        return base_model

    def build_full_model(self) -> tf.keras.Model:
        """Build complete model with base + custom layers"""
        base_model = self.build_base_model()
        
        inputs = keras.Input(shape=(*self.config.IMG_SIZE, 3))
        
        # Preprocessing: normalize pixel values
        x = keras.applications.resnet50.preprocess_input(inputs)
        
        # Base model (feature extraction)
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(
            len(self.config.CLASS_NAMES),
            activation='softmax'
        )(x)
        
        model = keras.Model(inputs, outputs)
        return model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Compile model with optimizer and loss function"""
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
