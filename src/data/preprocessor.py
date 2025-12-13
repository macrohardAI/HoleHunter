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

        # PENTING: HAPUS rescale=1./255
        # Kita biarkan raw pixel (0-255) masuk ke model,
        # karena Normalisasi akan dilakukan oleh layer 'Rescaling' di dalam model.
        train_datagen = ImageDataGenerator(
            # rescale=1./255,  <-- INI PENYEBAB MODEL COLLAPSE, JANGAN DIPAKAI
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False, # SARAN: Set False untuk jalan raya (aspal tidak pernah di langit)
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )

        # Validation juga jangan di-rescale
        val_test_datagen = ImageDataGenerator() # rescale dihapus

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
            # PENTING: Paksa urutan kelas sesuai Config agar bobot (Class Weights) valid
            classes=self.config.CLASS_NAMES
        )

        return data_generator

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        # Fungsi utilitas ini boleh tetap ada untuk debug visualisasi,
        # tapi tidak dipakai di pipeline training utama.
        return image / 255.0

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        return (image * 255).astype(np.uint8)