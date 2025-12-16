import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Config

class ModelBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build_base_model(self) -> tf.keras.Model:
        """Load pre-trained base model"""
        
        if self.config.BASE_MODEL == 'resnet50':
            base_model = keras.applications.ResNet50(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )

            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
                
        elif self.config.BASE_MODEL == 'mobilenetv2':
            # Alpha 1
            base_model = keras.applications.MobileNetV2(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet',
                alpha=1 # Alpha=1
            )
            # Freeze model
            base_model.trainable = False 
            
        else:
            raise ValueError(f"Unknown base model: {self.config.BASE_MODEL}")

        return base_model

    def build_full_model(self) -> tf.keras.Model:
        base_model = self.build_base_model()
        inputs = keras.Input(shape=(*self.config.IMG_SIZE, 3))

        # Dynamic Processing
        if self.config.BASE_MODEL == 'resnet50':
            x = keras.applications.resnet50.preprocess_input(inputs)
        elif self.config.BASE_MODEL == 'mobilenetv2':
            # layer rescaling
            x = layers.Rescaling(1./127.5, offset=-1)(inputs)
        else:
            x = inputs

        # Base model inference
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        # Head Architecture
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(0.2)(x) 


        # Output layer
        outputs = layers.Dense(
            len(self.config.CLASS_NAMES),
            activation='softmax'
        )(x)

        model = keras.Model(inputs, outputs)
        return model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE,
                # clipnorm for stability
                clipnorm=1.0 
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
