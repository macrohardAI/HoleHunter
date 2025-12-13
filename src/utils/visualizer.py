"""
Plotting training curves, confusion matrix, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_training_history(history: History, save_path: str = None) -> None:
        """Plot accuracy and loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        else:
            plt.show()

    @staticmethod
    def show_predictions(images: list, predictions: list, true_labels: list = None, 
                        class_names: list = None, cols: int = 4) -> None:
        """Display images with predictions."""
        num_images = len(images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (image, prediction) in enumerate(zip(images, predictions)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Display image
            ax.imshow(image)
            
            # Build title
            pred_class = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            title = f"Pred: {pred_class}\nConfidence: {confidence:.2%}"
            
            if true_labels and idx < len(true_labels):
                title = f"True: {true_labels[idx]}\n" + title
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
