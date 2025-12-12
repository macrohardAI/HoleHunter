"""
Plotting training curves, confusion matrix, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import History


class Visualizer:
    @staticmethod
    def plot_training_history(history: History, save_path: str) -> None:
        """Plot accuracy and loss curves."""
        raise NotImplementedError

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
        """Plot confusion matrix."""
        raise NotImplementedError

    @staticmethod
    def show_predictions(images: list, predictions: list) -> None:
        """Display image with predictions."""
        raise NotImplementedError
