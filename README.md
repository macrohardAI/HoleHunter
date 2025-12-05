# HoleHunter

An Introduction to AI project that uses Convolutional Neural Network (CNN) to classify road holes by using set of pictures.

## 🧩 Features

- Binary classification (hole/no hole)
- GPS metadata extraction from images
- Location data storage in SQLite database

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

## 🔧 Usage

```bash
# Train model
python src/models/train.py

# Run prediction
python app/predict.py --image path/to/image.jpg
```

## 📂 Project Structure

```
HoleHunter/
│
├── data/
│   ├── raw/                    # Original, unprocessed images
│   │   ├── hole/
│   │   └── no_hole/
│   ├── processed/              # Preprocessed images (resized, augmented)
│   │   ├── train/
│   │   │   ├── hole/
│   │   │   └── no_hole/
│   │   ├── validation/
│   │   │   ├── hole/
│   │   │   └── no_hole/
│   │   └── test/
│   │       ├── hole/
│   │       └── no_hole/
│   └── sample/                 # Sample images for testing/demo
│
├── models/
│   ├── saved_models/           # Trained model files (.h5, .keras)
│   └── checkpoints/            # Training checkpoints
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset loading and preprocessing
│   │   └── augmentation.py     # Data augmentation functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py        # CNN architecture definition
│   │   └── train.py            # Training script
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metadata.py         # EXIF/GPS extraction
│   │   └── visualization.py    # Plotting and visualization
│   └── database/
│       ├── __init__.py
│       ├── db_manager.py       # Database operations
│       └── schema.sql          # Database schema
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_metadata.py
│
├── scripts/
│   ├── download_dataset.py     # Script to download/prepare dataset
│   ├── preprocess_data.py      # Preprocessing pipeline
│   └── evaluate_model.py       # Model evaluation
│
├── database/
│   └── holes.db                # SQLite database (gitignored)
│
├── app/
│   └── predict.py              # Main application for prediction
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
├── .python-version             # Python version info
├── config.py                   # Configuration parameters
└── setup.py                    # Package installation (optional)
```
