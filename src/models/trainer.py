"""
Training loop, callbacks, saving models.
"""
from keras.src.callbacks import History
import model_builder as mb


class Trainer:
    def __init__(self):
        pass

    def setup_callbacks(self) -> list:
        """Create ModelCheckpoint, EarlyStopping, etc."""
        raise NotImplementedError

    def train(self, train_gen, val_gen) -> History:
        """Main training loop"""
        raise NotImplementedError

    def save_model(self, path: str) -> None:
        """Save trained model"""
        raise NotImplementedError
