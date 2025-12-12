"""
Read images and returning as arrays
"""
import numpy as np


class ImageLoader:
    def load_image(self, path: str) -> np.ndarray:
        """Load single image"""
        raise NotImplementedError

    def load_dataset(self, directory: str) -> tuple:
        """Load all images from a directory"""
        raise NotImplementedError

    def count_images(self, directory: str) -> dict:
        """Count images per class"""
        raise NotImplementedError
