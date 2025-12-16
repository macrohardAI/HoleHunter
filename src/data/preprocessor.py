"""
Data preprocessing utilities
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import Config

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def create_data_generators(self):
        """Create ImageDataGenerator for train and validation"""

        train_datagen = ImageDataGenerator(
            # rescale=1./255,
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        # Validation generator
        val_test_datagen = ImageDataGenerator()

        return train_datagen, val_test_datagen

    def load_data_from_directory(self, directory, split_name='train'):
        """Load data using flow_from_directory"""
        train_datagen, val_test_datagen = self.create_data_generators()

        if split_name in ['train']:
            generator = train_datagen
        else:
            generator = val_test_datagen

        data_generator = generator.flow_from_directory(
            directory,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=(split_name == 'train'),
            classes=self.config.CLASS_NAMES
        )

        return data_generator

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        
        return image / 255.0

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        return (image * 255).astype(np.uint8)