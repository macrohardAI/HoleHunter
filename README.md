# HoleHunter2 - Sistem Deteksi dan Pemetaan Jalan Berlubang di Kalimantan

Sistem deteksi otomatis untuk mengidentifikasi dan memetakan kerusakan jalan menggunakan deep learning dengan MobileNetV2. Proyek ini mengklasifikasikan tingkat kerusakan jalan menjadi tiga kategori: Normal, Medium, dan Severe.

## Tim Pengembang

| Nama | NIM |
|Data|Data|
| Andi Naufal Nurfadhil | 11241014 |
| Kevin Jonathan Wijaya | 11241040 |
| Muhammad Aditya Putra | 11241050 |
| Rana Afifah Dzikro | 11241076 |
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

```bash
git clone <repository-url>
cd holehunter2