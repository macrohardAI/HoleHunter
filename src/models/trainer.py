"""
Training loop, callbacks, saving models.
"""
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from keras.src.callbacks import History

from config import Config
from src.models.model_builder import ModelBuilder


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.history = None

    def setup_callbacks(self, model_dir='./models') -> list:
        """Create ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Save best model based on validation accuracy
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # Stop training if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate if validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            ),

            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1
            )
        ]

        return callbacks

    def train(
        self,
        train_generator,
        val_generator,
        model_dir='./models'
    ) -> History:
        """Main training loop"""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

        # Build and compile model
        builder = ModelBuilder(self.config)
        self.model = builder.build_full_model()
        self.model = builder.compile_model(self.model)

        print(self.model.summary())

        # Setup callbacks
        callbacks = self.setup_callbacks(model_dir)

        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=self.config.EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return self.history

    def save_model(self, path: str = './models/trained_model.keras') -> None:
        """Save trained model in .keras format (Keras 3 compatible)"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure path ends with .keras
        if not path.endswith('.keras'):
            path = path + '.keras'

        self.model.save(path)
        print(f"✅ Model saved to {path}")
