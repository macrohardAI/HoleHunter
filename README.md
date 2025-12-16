# HoleHunter2

## Sistem Deteksi dan Pemetaan Jalan Berlubang di Kalimantan

Sistem deteksi otomatis untuk mengidentifikasi dan memetakan kerusakan jalan menggunakan deep learning berbasis **MobileNetV2**. Proyek ini mengklasifikasikan tingkat kerusakan jalan menjadi tiga kategori: **Normal**, **Medium**, dan **Severe**.

---

## Tim Pengembang

| Nama                   | NIM      |
| ---------------------- | -------- |
| Andi Naufal Nurfadhil  | 11241014 |
| Kevin Jonathan Wijaya  | 11241040 |
| Muhammad Aditya Putra  | 11241050 |
| Rana Afifah Dzikro     | 11241076 |
| Sulthan Farizan Fawwaz | 11241080 |

---

## Daftar Isi

* [Tentang Proyek](#tentang-proyek)
* [Fitur Utama](#fitur-utama)
* [Teknologi](#teknologi)
* [Instalasi](#instalasi)
* [Penggunaan](#penggunaan)
* [Struktur Proyek](#struktur-proyek)
* [Konfigurasi](#konfigurasi)
* [Hasil](#hasil)
* [Dokumentasi](#dokumentasi)

---

## Tentang Proyek

**HoleHunter2** adalah sistem berbasis kecerdasan buatan yang dirancang untuk mendeteksi dan memetakan kondisi jalan di Kalimantan. Sistem ini menggunakan arsitektur **MobileNetV2** yang dioptimalkan untuk klasifikasi gambar dengan tiga tingkat kerusakan:

* **Normal** — Jalan dalam kondisi baik
* **Medium** — Kerusakan tingkat sedang
* **Severe** — Kerusakan parah dan berbahaya

Sistem mendukung pemrosesan gambar tunggal maupun batch, serta menghasilkan peta interaktif berbasis HTML untuk visualisasi lokasi kerusakan.

---

## Fitur Utama

* Deteksi otomatis tingkat kerusakan jalan berbasis deep learning
* Pemetaan interaktif menggunakan peta HTML
* Data augmentation untuk meningkatkan generalisasi model
* Evaluasi model lengkap (confusion matrix dan metrik performa)
* Prediksi batch untuk efisiensi pemrosesan
* Checkpoint model terbaik selama training

---

## Teknologi

### Framework dan Library

* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* NumPy dan Pandas
* Pillow
* Matplotlib
* Seaborn
* Folium

### Konfigurasi Model

* Input Size: 224 × 224 piksel
* Batch Size: 32
* Epochs: 80
* Learning Rate: 0.001
* Class Weights: Balanced

---

## Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd holehunter2
```

### 2. Virtual Environment (Direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Persiapan Dataset

Struktur folder dataset:

```text
data/
├── raw/
│   ├── normal/
│   ├── medium/
│   └── severe/
└── processed/
    ├── train/
    ├── validation/
    └── test/
```

Letakkan dataset gambar mentah pada folder `data/raw/` sesuai kelasnya.

---

## Penggunaan

### 1. Persiapan Dataset

```bash
python data_preparation.py
```

### 2. Data Augmentation

```bash
python src/data/augmentation.py data/processed/train --variants <jumlah>
```

Setelah augmentasi, pastikan struktur folder training sudah konsisten sebelum proses training dimulai.

### 3. Training Model

```bash
python train.py
```

Output training:

* `models/trained_model.keras`
* `models/best_model.keras`
* `models/training_history.png`
* `models/confusion_matrix.png`

### 4. Prediksi

**Prediksi satu gambar**

```bash
python predict.py --image path/to/image.jpg
```

**Prediksi batch (folder)**

```bash
python predict.py --batch path/to/folder
```

**Custom model**

```bash
python predict.py --model models/best_model.keras --image test.jpg
```

Output prediksi:

* Kelas dan confidence score
* File peta interaktif: `laporan_peta_kerusakan.html`
* Visualisasi hasil prediksi

---

## Struktur Proyek

```text
holehunter2/
├── config.py
├── data_preparation.py
├── train.py
├── predict.py
├── test.py
├── requirements.txt
│
├── src/
│   ├── data/
│   │   ├── augmentation.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── model_builder.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── helpers/
│       ├── visualizer.py
│       └── map_generator.py
│
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── docs/
    └── REPORT.md
```

---

## Konfigurasi

Edit file `config.py`:

```python
class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 80
    LEARNING_RATE = 0.001

    BASE_MODEL = "mobilenetv2"
    CLASS_NAMES = ["medium", "normal", "severe"]
    CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.5}
```

---

## Hasil

Model menunjukkan performa yang baik dalam mengklasifikasikan tingkat kerusakan jalan. Evaluasi lengkap tersedia dalam file hasil training.

Output utama:

* Grafik akurasi dan loss
* Confusion matrix
* Peta interaktif lokasi kerusakan

---

## Dokumentasi

Dokumentasi teknis lengkap tersedia pada:

* `docs/REPORT.md`

---

## Lisensi

Proyek ini dikembangkan untuk keperluan akademik.

---

## Kontribusi

Saran dan kontribusi dipersilakan melalui issue atau pull request.

---

## Kontak

Silakan hubungi tim pengembang untuk diskusi dan pertanyaan lanjutan.