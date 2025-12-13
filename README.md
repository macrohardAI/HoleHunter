# 🕳️ HOLE HUNTER - Pothole Detection System

Sistem deteksi jalan berlubang menggunakan Deep Learning (CNN dengan Transfer Learning).

## 📋 Daftar Isi
1. [Instalasi](#instalasi)
2. [Struktur Proyek](#struktur-proyek)
3. [Persiapan Data](#persiapan-data)
4. [Training Model](#training-model)
5. [Prediksi](#prediksi)
6. [API Integration](#api-integration)

---

## 🚀 Instalasi

### Requirements
- Python 3.8+
- TensorFlow 2.20+
- CUDA (Optional, untuk training lebih cepat)

### Setup

1. **Install Python Dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. **Verifikasi Instalasi**
\`\`\`bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
\`\`\`

---

## 📁 Struktur Proyek

\`\`\`
hole-hunter/
├── data/
│   ├── raw/                          # Data asli
│   │   ├── hole/                     # Gambar jalan berlubang
│   │   └── no_hole/                  # Gambar jalan normal
│   └── processed/                    # Data terproses (setelah split)
│       ├── train/
│       │   ├── hole/
│       │   └── no_hole/
│       ├── validation/
│       │   ├── hole/
│       │   └── no_hole/
│       └── test/
│           ├── hole/
│           └── no_hole/
├── src/
│   ├── data/
│   │   ├── loader.py                 # Load gambar dari directory
│   │   └── preprocessor.py           # Data augmentation & preprocessing
│   ├── models/
│   │   ├── model_builder.py          # Build CNN architecture
│   │   ├── trainer.py                # Training loop
│   │   └── evaluator.py              # Model evaluation
│   ├── database/
│   │   └── db_manager.py             # SQLite untuk menyimpan detections
│   └── utils/
│       └── visualizer.py             # Plotting & visualization
├── models/                           # Simpan trained models
│   ├── trained_model.h5             # Model final
│   ├── best_model.h5                # Best checkpoint
│   ├── training_history.png         # Accuracy/Loss curves
│   └── confusion_matrix.png         # Evaluation results
├── config.py                         # Konfigurasi global
├── data_preparation.py              # Script untuk split data
├── train.py                         # Main training script
├── predict.py                       # Inference script
└── requirements.txt
\`\`\`

---

## 📊 Persiapan Data

### Step 1: Siapkan Dataset

Kumpulkan gambar jalan dan letakkan di:
- `data/raw/hole/` - Gambar jalan berlubang
- `data/raw/no_hole/` - Gambar jalan normal

\`\`\`
data/raw/
├── hole/
│   ├── pothole_1.jpg
│   ├── pothole_2.jpg
│   └── ...
└── no_hole/
    ├── normal_road_1.jpg
    ├── normal_road_2.jpg
    └── ...
\`\`\`

### Step 2: Generate Sample Data (Opsional - untuk testing)

Jika belum punya data real, buat dummy dataset:

\`\`\`bash
python data_preparation.py --create-sample --raw-dir data/raw
\`\`\`

Ini akan membuat 10 sample images per class untuk testing pipeline.

### Step 3: Split Data

Split dataset menjadi train (70%), validation (15%), dan test (15%):

\`\`\`bash
python data_preparation.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --train-split 0.7 \
    --val-split 0.15 \
    --test-split 0.15 \
    --image-size 224
\`\`\`

### Step 4: Verifikasi Data

\`\`\`bash
python data_preparation.py --verify --output-dir data/processed
\`\`\`

Output akan menampilkan:
\`\`\`
TRAIN:
  hole: 100 images
  no_hole: 100 images
  Total: 200

VALIDATION:
  hole: 21 images
  no_hole: 21 images
  Total: 42

TEST:
  hole: 22 images
  no_hole: 22 images
  Total: 44

TOTAL IMAGES: 286
\`\`\`

---

## 🧠 Training Model

### Konfigurasi (config.py)

Sesuaikan parameter sesuai kebutuhan:

\`\`\`python
class Config:
    # Paths
    DATA_DIR: str = "./data"           # Data directory
    MODEL_DIR: str = "./models"        # Model save directory

    # Model parameters
    IMG_SIZE: tuple = (224, 224)       # Input image size
    BATCH_SIZE: int = 32               # Batch size per iteration
    EPOCHS: int = 50                   # Training epochs
    LEARNING_RATE: float = 0.001       # Optimizer learning rate

    # Model selection: 'resnet50' atau 'mobilenetv2'
    BASE_MODEL: str = "resnet50"       # Transfer learning base

    # Classes
    CLASS_NAMES: list = ['no_hole', 'hole']
\`\`\`

### Run Training

\`\`\`bash
python train.py
\`\`\`

Output:
\`\`\`
============================================================
HOLE HUNTER - POTHOLE DETECTION MODEL TRAINING
============================================================

📋 Configuration:
  - Base Model: resnet50
  - Image Size: (224, 224)
  - Batch Size: 32
  - Epochs: 50
  - Learning Rate: 0.001
  - Classes: ['no_hole', 'hole']

✅ Data directory found: ./data/processed/

📊 Creating data generators...
Found 200 training images belonging to 2 classes.
Found 42 validation images belonging to 2 classes.
Found 44 test images belonging to 2 classes.

🚀 Starting training...
Epoch 1/50
...
Epoch 50/50 [==============================] - 120s 2s/step
  loss: 0.1234 - accuracy: 0.9456 - val_loss: 0.1567 - val_accuracy: 0.9234

🔍 Evaluating model...
Accuracy: 0.9234
Precision: 0.9145
Recall: 0.9345
F1: 0.9244

📈 Generating visualizations...
✅ Training history saved to models/training_history.png
✅ Confusion matrix saved to models/confusion_matrix.png

✅ TRAINING COMPLETE!
\`\`\`

---

## 🔮 Prediksi (Inference)

### Single Image Prediction

\`\`\`bash
python predict.py --image data/test_image.jpg
\`\`\`

Output:
\`\`\`
🔍 Loading model from: models/trained_model.h5
📷 Predicting on: data/test_image.jpg

==================================================
PREDICTION RESULT
==================================================
Image: test_image.jpg
Classification: hole
Confidence: 94.23%

Probabilities:
  - no_hole: 0.0577
  - hole: 0.9423
==================================================
\`\`\`

### Batch Prediction (Multiple Images)

\`\`\`bash
python predict.py --batch data/test_images/
\`\`\`

Output:
\`\`\`
🔍 Loading model from: models/trained_model.h5
📷 Found 5 images to predict
  ✅ image_1.jpg: hole (92.45%)
  ✅ image_2.jpg: no_hole (95.67%)
  ✅ image_3.jpg: hole (87.23%)
  ✅ image_4.jpg: no_hole (93.12%)
  ✅ image_5.jpg: hole (89.34%)
\`\`\`

### Custom Model Path

\`\`\`bash
python predict.py --model models/best_model.h5 --image data/test.jpg
\`\`\`

---

## 🗄️ Database Integration

Model predictions dapat disimpan ke SQLite database:

\`\`\`python
from src.database.db_manager import DatabaseManager

# Inisialisasi database
db = DatabaseManager('potholes.db')

# Simpan detection
detection_id = db.insert_detection(
    image_path='data/test.jpg',
    class_name='hole',
    confidence=0.9423,
    latitude=-6.2088,      # Jakarta coordinates (example)
    longitude=106.8456
)

# Ambil semua detections
all_detections = db.get_all_detections()

# Cari detections di sekitar lokasi (radius 1 km)
nearby = db.get_detections_by_location(
    latitude=-6.2088,
    longitude=106.8456,
    radius_km=1.0
)

# Dapatkan statistik
stats = db.get_statistics()
print(f"Total detections: {stats['total_detections']}")
print(f"Holes found: {stats['holes_found']}")
print(f"Hole percentage: {stats['hole_percentage']:.2f}%")

# Tutup database
db.close()
\`\`\`

---

## 🌐 API Integration (Next.js)

Buat API endpoint untuk model inference:

\`\`\`python
# app/api/predict/route.py
import next from 'next/server'
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'models' / 'trained_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

export async function POST(request: Request) {
    try:
        const formData = await request.formData()
        const file = formData.get('image')
        
        # Process image dan predict
        # Return JSON with results
        
        return Response.json({
            class: 'hole',
            confidence: 0.9423,
            probabilities: {
                'no_hole': 0.0577,
                'hole': 0.9423
            }
        })
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 })
    }
}
\`\`\`

---

## 📈 Model Performance

Metrik evaluasi yang digunakan:

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | % prediksi benar dari total |
| **Precision** | % prediksi positif yang benar |
| **Recall** | % positif aktual yang terdeteksi |
| **F1 Score** | Harmonic mean precision & recall |

Confusion Matrix:
\`\`\`
                 Predicted
               no_hole  hole
True  no_hole   [ ]     [ ]
      hole      [ ]     [ ]
\`\`\`

---

## 🔧 Tips & Troubleshooting

### 1. Training terlalu lambat?
- Gunakan GPU: Install CUDA
- Reduce `BATCH_SIZE` jika out of memory
- Reduce `EPOCHS` untuk testing cepat

### 2. Memory error?
\`\`\`python
# config.py
BATCH_SIZE = 16  # Reduce dari 32
IMG_SIZE = (128, 128)  # Reduce dari (224, 224)
\`\`\`

### 3. Model overfitting?
- Tambah data augmentation
- Increase dropout rate
- Reduce model complexity (MobileNetV2)

### 4. Prediksi tidak akurat?
- Verifikasi data training quality
- Retrain dengan lebih banyak epochs
- Gunakan data yang lebih representatif

---

## 📚 Architecture Details

### Transfer Learning dengan ResNet50

\`\`\`
Input (224x224x3)
        ↓
ResNet50 (Pre-trained on ImageNet)
        ↓
Global Average Pooling
        ↓
Dense(256) + ReLU + Dropout(0.3)
        ↓
Dense(128) + ReLU + Dropout(0.3)
        ↓
Dense(2) + Softmax  → [no_hole, hole]
\`\`\`

### Data Augmentation

- Random rotation (20°)
- Width/Height shift (20%)
- Shear (20%)
- Zoom (20%)
- Horizontal flip

---

## 📞 Support & Kontribusi

Jika ada pertanyaan atau issue:
1. Check log files di `models/logs/`
2. Verifikasi data di `data/processed/`
3. Cek config di `config.py`

---

**Last Updated:** 13 December 2024  
**Version:** 1.0.0
