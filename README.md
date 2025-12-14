# HoleHunter2 - Sistem Deteksi dan Pemetaan Jalan Berlubang di Kalimantan

Sistem deteksi otomatis untuk mengidentifikasi dan memetakan kerusakan jalan menggunakan deep learning dengan MobileNetV2. Proyek ini mengklasifikasikan tingkat kerusakan jalan menjadi tiga kategori: Normal, Medium, dan Severe.

## Tim Pengembang

| Nama                   | NIM      |
|------------------------|----------|
| Andi Naufal Nurfadhil  | 11241014 |
| Kevin Jonathan Wijaya  | 11241040 |
| Muhammad Aditya Putra  | 11241050 |
| Rana Afifah Dzikro     | 11241076 |
| Sulthan Farizan Fawwaz | 11241080 |

---

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Teknologi](#teknologi)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Konfigurasi](#konfigurasi)
- [Hasil](#hasil)
- [Dokumentasi](#dokumentasi)

---

## Tentang Proyek

HoleHunter2 adalah sistem berbasis kecerdasan buatan yang dirancang untuk mendeteksi dan memetakan kondisi jalan di Kalimantan. Sistem ini menggunakan arsitektur MobileNetV2 yang dioptimalkan untuk klasifikasi gambar dengan tiga kategori tingkat kerusakan:

- **Normal**: Jalan dalam kondisi baik
- **Medium**: Kerusakan jalan tingkat sedang
- **Severe**: Kerusakan jalan parah yang memerlukan perhatian segera

Sistem ini dapat memproses gambar individual maupun batch, dan secara otomatis menghasilkan peta interaktif untuk visualisasi lokasi kerusakan jalan.

---

## Fitur Utama

- **Deteksi Otomatis**: Klasifikasi tingkat kerusakan jalan menggunakan deep learning
- **Pemetaan Interaktif**: Visualisasi lokasi kerusakan pada peta HTML
- **Data Augmentation**: Meningkatkan variasi dataset untuk akurasi lebih baik
- **Model Evaluation**: Confusion matrix dan metrik performa lengkap
- **Batch Processing**: Prediksi massal untuk efisiensi
- **Checkpoint System**: Menyimpan model terbaik selama training

---

## Teknologi

### Framework & Library
- **TensorFlow/Keras**: Deep learning framework
- **MobileNetV2**: Arsitektur model untuk transfer learning
- **NumPy & Pandas**: Manipulasi data
- **Pillow**: Pemrosesan gambar
- **Matplotlib & Seaborn**: Visualisasi
- **Folium**: Pembuatan peta interaktif

### Konfigurasi Model
- Input Size: 224x224 pixels
- Batch Size: 32
- Epochs: 80
- Learning Rate: 0.001
- Class Weights: Balanced untuk handling imbalanced data

---

## Instalasi

### 1. Clone Repository

\`\`\`bash
git clone <repository-url>
cd holehunter2
\`\`\`

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)

\`\`\`bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
\`\`\`

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Persiapan Data

Struktur folder data yang dibutuhkan:

\`\`\`
data/
├── raw/
│   ├── medium/
│   ├── normal/
│   └── severe/
└── processed/  # Akan dibuat otomatis
    ├── train/
    ├── validation/
    └── test/
\`\`\`

Letakkan dataset gambar di folder `data/raw/` sesuai kategorinya.

---

## Penggunaan

### 1. Persiapan Dataset

Pisahkan data menjadi train, validation, dan test set:

\`\`\`bash
python data_preparation.py
\`\`\`

### 2. Augment dataset

Perbanyak dataset

\`\`\`bash
python src/data/augmentation.py data/processed/train --variants <Sesuaikan jumlah yang akan di augment>
\`\`\`

Setelah itu jangan lupa untuk rename folder augment dan folder train aslinya agar model train dari data augment

### 3. Training Model

Latih model dengan dataset yang telah dipersiapkan:

\`\`\`bash
python train.py
\`\`\`

Output:
- `models/trained_model.keras`: Model hasil training
- `models/best_model.keras`: Model dengan performa terbaik
- `models/training_history.png`: Grafik akurasi dan loss
- `models/confusion_matrix.png`: Analisis prediksi

### 4. Prediksi

**Prediksi Gambar Tunggal:**

\`\`\`bash
python predict.py --image path/to/image.jpg
\`\`\`

**Prediksi Batch (Folder):**

\`\`\`bash
python predict.py --batch path/to/image/folder
\`\`\`

**Custom Model Path:**

\`\`\`bash
python predict.py --model models/best_model.keras --image test.jpg
\`\`\`

Output:
- Klasifikasi dan confidence score
- Peta HTML interaktif: `laporan_peta_kerusakan.html`
- Visualisasi gambar dengan prediksi

---

## Struktur Proyek

\`\`\`
holehunter2/
├── config.py                 # Konfigurasi global
├── data_preparation.py       # Script pemrosesan dataset
├── train.py                  # Script training model
├── predict.py                # Script inferensi/prediksi
├── test.py                   # Unit testing
├── requirements.txt          # Dependencies
│
├── src/
│   ├── data/
│   │   ├── augmentation.py   # Data augmentation
│   │   ├── loader.py         # Data loading utilities
│   │   └── preprocessor.py   # Preprocessing pipeline
│   │
│   ├── models/
│   │   ├── model_builder.py  # Arsitektur model
│   │   ├── trainer.py        # Training logic
│   │   └── evaluator.py      # Evaluasi dan prediksi
│   │
│   └── helpers/
│       ├── visualizer.py     # Plotting dan visualisasi
│       └── map_generator.py  # Generator peta HTML
│
├── data/
│   ├── raw/                  # Dataset asli
│   └── processed/            # Dataset terproses
│
├── models/                   # Saved models dan hasil
│
└── docs/
    └── REPORT.md             # Dokumentasi lengkap
\`\`\`

---

## Konfigurasi

Edit file `config.py` untuk menyesuaikan parameter:

\`\`\`python
class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 80
    LEARNING_RATE = 0.001
    
    BASE_MODEL = "mobilenetv2"
    CLASS_NAMES = ['medium', 'normal', 'severe']
    CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.5}
\`\`\`

---

## Hasil

### Metrik Performa
Model mencapai performa yang baik dalam mengklasifikasikan tingkat kerusakan jalan dengan akurasi tinggi pada dataset test. Detail lengkap tersedia di file `confusion_matrix.png` setelah training.

### Output
- **Training History**: Grafik loss dan akurasi untuk monitoring proses training
- **Confusion Matrix**: Analisis detail prediksi per kelas
- **Interactive Map**: Peta HTML dengan marker lokasi kerusakan jalan

---

## Dokumentasi

Untuk informasi lebih detail mengenai metodologi, implementasi, dan hasil eksperimen, silakan lihat:

- [REPORT.md](docs/REPORT.md) - Laporan teknis lengkap

---

## Lisensi

Proyek ini dikembangkan untuk keperluan akademis.

---

## Kontribusi

Kontribusi dan saran sangat diterima. Silakan buat issue atau pull request untuk perbaikan.

---

## Kontak

Untuk pertanyaan atau diskusi lebih lanjut, silakan hubungi tim pengembang melalui informasi kontak yang tersedia.