"""
Read images and returning as arrays
"""
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf


class ImageLoader:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size

    def load_image(self, path: str) -> np.ndarray:
        """Load single image and convert to array"""
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            return np.array(img) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def load_dataset(self, directory: str) -> tuple:
        """Load all images from directory structure: dir/class/image.jpg"""
        images = []
        labels = []
        class_names = {}
        class_idx = 0

        data_path = Path(directory)
        
        # Get all class folders
        for class_folder in sorted(data_path.iterdir()):
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            class_names[class_idx] = class_name
            
            print(f"Loading {class_name}...")
            
            # Get all images in class folder
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    img_array = self.load_image(str(img_path))
                    if img_array is not None:
                        images.append(img_array)
                        labels.append(class_idx)
            
            class_idx += 1

        return np.array(images), np.array(labels), class_names

    def count_images(self, directory: str) -> dict:
        """Count images per class"""
        counts = {}
        data_path = Path(directory)
        
        for class_folder in data_path.iterdir():
            if class_folder.is_dir():
                count = len(list(class_folder.glob('*')))
                counts[class_folder.name] = count
        
        return counts
