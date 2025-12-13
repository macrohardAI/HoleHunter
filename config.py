"""
The file name pretty much self-explanatory
"""


class Config:
    # Paths
    DATA_DIR: str = "./data"
    MODEL_DIR: str = "./models"

    # Model parameters
    IMG_SIZE: tuple = (224, 224)
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001

    # Model selection: 'resnet50', 'mobilenetv2'
    BASE_MODEL: str = "mobilenetv2"

    # Classes
    CLASS_NAMES: list = ['severe', 'medium', 'normal']