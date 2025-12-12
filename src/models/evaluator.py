"""
Testing model, generating metrics
"""
import numpy as np
import tensorflow as tf


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self) -> dict:
        """Calculate accuracy, precision, recall, F1"""
        raise NotImplementedError

    def confusion_matrix(self, test_gen) -> np.ndarray:
        """Generate confusion matrix"""
        raise NotImplementedError

    def predict_single(self, image_path: str) -> dict:
        """Predict on a single image"""
        raise NotImplementedError
