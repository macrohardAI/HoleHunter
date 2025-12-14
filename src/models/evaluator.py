"""
Testing model, generating metrics
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from PIL import Image


class ModelEvaluator:
    def __init__(self, model: tf.keras.Model, config):
        self.model = model
        self.config = config

    def evaluate(self, test_generator) -> dict:
        """Calculate accuracy, precision, recall, F1"""
        print("=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Get predictions and true labels
        predictions = self.model.predict(test_generator)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = test_generator.labels
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted'),
            'recall': recall_score(true_labels, pred_labels, average='weighted'),
            'f1': f1_score(true_labels, pred_labels, average='weighted')
        }
        
        # Print detailed report
        print("\nClassification Report:")
        print(classification_report(
            true_labels,
            pred_labels,
            target_names=self.config.CLASS_NAMES
        ))
        
        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics

    def confusion_matrix(self, test_generator) -> np.ndarray:
        """Generate confusion matrix"""
        predictions = self.model.predict(test_generator)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = test_generator.labels
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        print("\nConfusion Matrix:")
        print(cm)
        
        return cm

    def predict_single(self, image_path: str) -> dict:
        """Predict on a single image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.config.IMG_SIZE, Image.Resampling.LANCZOS)

            # --- PERBAIKAN DI SINI ---
            img_array = np.array(img)  # JANGAN DIBAGI 255.0
            # -------------------------

            img_array = np.expand_dims(img_array, axis=0)

            # print(f"Debug Min: {img_array.min()}, Max: {img_array.max()}")

            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            confidence = np.max(prediction)
            class_idx = np.argmax(prediction)
            class_name = self.config.CLASS_NAMES[class_idx]

            result = {
                'class': class_name,
                'confidence': float(confidence),
                'probabilities': {
                    self.config.CLASS_NAMES[i]: float(prediction[0][i])
                    for i in range(len(self.config.CLASS_NAMES))
                }
            }

            return result

        except Exception as e:
            print(f"Error predicting on {image_path}: {e}")
            return None
