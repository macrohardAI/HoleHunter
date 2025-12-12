"""
For creating CNN model
"""
import tensorflow as tf

from config import Config


class ModelBuilder:
    def __init__(self, config: Config):
        pass

    def build_base_model(self) -> tf.keras.Model:
        """Load pre-trained base model"""
        raise NotImplementedError
