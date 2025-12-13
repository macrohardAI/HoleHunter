"""
The file name pretty much self-explanatory
"""


class Config:
    # Paths
    DATA_DIR: str = "./data"
    MODEL_DIR: str = "./models"

    IMG_SIZE: tuple = (224, 224)
    BATCH_SIZE: int = 32  # Smaller batch size for more stable training
    EPOCHS: int = 80  # More epochs to allow better convergence
    LEARNING_RATE: float = 0.001  # Lower learning rate for more stable training

    BASE_MODEL: str = "mobilenetv2"

    # Classes
    CLASS_NAMES: list = ['medium', 'normal', 'severe']

    # Increase weight for severe class since it's being misclassified
    CLASS_WEIGHTS: dict = {0: 1.0, 1: 1.5, 2: 2.5}  # severe, medium, normal