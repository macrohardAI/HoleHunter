# Sistem Mapping Jalan Berlubang di Kalimantan

<!-- Insert description -->

## 1st Group

| Name                   | Student ID |
|------------------------|------------|
| Andi Naufal Nurfadhil  | 11241014   |
| Kevin Jonathan Wijaya  | 11241040   |
| Muhammad Aditya Putra  | 11241050   |
| Rana Afifah Dzikro     | 11241076   |
| Sulthan Farizan Fawwaz | 11241080   |

## Abstract

Jalan berlubang dapat mengganggu kenyamanan pengendara di lalu lintas. Selain itu, jalan berlubang juga berpotensi membahayakan pengendara terutama para pengemudi motor di Kalimantan yang banyak melalui jalur darat untuk bepergian antar kota yang pastinya memerlukan jalan yang layak untuk dilewati. Karena adanya masalah tersebut, mendorong penulis untuk menciptakan suatu sistem yang mampu mengklasifikasikan jalan berlubang melalui foto yang memiliki metadata dengan menerapkan metode Convolutional Neural Network (CNN). 

**Keywords:** Deep Learning, Transfer Learning, MobileNetV2, Road Plothole Detection, Image Classification

## Methods

Klasifikasi jalan berlubang menggunakan teknik klasifikasi gambar dengan lebih dari satu kelas menggunakan gambar input berukuran 224 px x 224 px RGB. Sistem  ini menggunakan Transfer Learning dengan MobileNetV2 sebagai base model. HoleHunter bertujuan untuk memprediksi tingkat keparahan kerusakan jalan dan  mengklasifikasikannya ke dalam salah satu dari tiga kelas yaitu:
- Medium (Rusak ringan)
- Normal
- Severe (Rusak berat)

## **2.1 MobileNetV2**
Kami menggunakan transfer learning dengan MobileNetV2 sebagai base model karena efisiensinya untuk deployments  mobile dan akurasi yang  lebih tinggi dari model lain.

### **1. Input (224×224×3)**

### **2. Rescalling Normalization**
Input gambar di normalisasi ke range [-1,1] sesuai dengan kebutuhan MobileNetV2.

### **`model_builder.py`**
```python
# Line 47-48
x = layers.Rescaling(1./127.5, offset=-1)(inputs)
```
$$x_{\text{normalized}} = \frac{x}{127.5} - 1$$

### **3. MobileNetV2 Base (32_224_f)**

MobileNetV2 mengubah gambar menjadi fitur-fitur penting seperti tekstur jalan atau luubang.

### **4.Global Average Pooling**

Mengubah feature map (7 x 7 x 1280) menjadi vektor 1D (1280)

### **`model_builder.py`**
```python
# Line 56
x = layers.GlobalAveragePooling2D()(x)
```

$$\text{GAP}(x) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,k}$$

### **5. Dense (Kernel 1280x100, bias 100)**

### **6. Activation**

### **`model_builder.py`**
```python
# Line 68-69
x = layers.Dense(100, activation='relu')(x)
```

$$y = \text{ReLU}(W \cdot x + b)$$
$$\text{ReLU}(z) = \max(0, z)$$

di mana:
- $W \in \mathbb{R}^{100 \times 1280}$
- $x \in \mathbb{R}^{1280}$
- $b \in \mathbb{R}^{100}$
- $y \in \mathbb{R}^{100}$

### **7. Dropout**

### **8. Dense (Kernel 100x3, bias 3)**

### **`model_builder.py`**
```python
# Line 74-77
outputs = layers.Dense(
    len(self.config.CLASS_NAMES),  # 3 classes
    activation='softmax'
)(x)
```

$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}$$

di mana:
- $z_k = W_k \cdot x + b_k$
- $C = 3$ (jumlah kelas)
- $\sum_{k=1}^{3} P(y=k|x) = 1$

### **9. Dense_1**

Output terakhir model

## **2.2 Loss Function**

Dataset jalan berlubang yang kami gunakan memiliki ketidakseimbangan kelas. Kami mengatasi masalah ini dengan menggunakan weighted loss function dan memberikan bobot tertinggi untuk kelas severe (2.5) untuk meningkatkan sensitivitas. 

### **`model_builder.py`**
```python
# Line 84-86
loss='categorical_crossentropy'
```

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} w_{ij} \cdot y_{ij} \cdot \log(\hat{y}_{ij})$$

di mana:
- $N$ = jumlah sampel dalam batch
- $C = 3$ = jumlah kelas
- $y_{ij}$ = label ground truth (one-hot encoded)
- $\hat{y}_{ij}$ = probabilitas prediksi dari softmax
- $w_j$ = class weight untuk kelas $j$


## **2.3. Optimizer**

### **`model_builder.py`**
```python
# Line 81-86
optimizer=keras.optimizers.Adam(
    learning_rate=self.config.LEARNING_RATE,  # 0.001
    clipnorm=1.0
)
```

**a. Compute gradient**

$$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$$

**b. Update biased first moment (momentum)**

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

**c. Update biased second moment (variance)**

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

**c. Bias correction**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**d. Update parameters**

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default values di Keras:**
- $\alpha = 0.001$ (learning rate dari config)
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-7}$

## **2.4 Evaluation Metrics**

### **`evaluator.py`**

#### **Kode:**
```python
# Line 29-32
metrics = {
    'accuracy': accuracy_score(true_labels, pred_labels),
    'precision': precision_score(true_labels, pred_labels, average='weighted'),
    'recall': recall_score(true_labels, pred_labels, average='weighted'),
    'f1': f1_score(true_labels, pred_labels, average='weighted')
}
```

**Confusion Matrix (3×3):**

$$CM = \begin{bmatrix}
TP_0 & E_{01} & E_{02} \\
E_{10} & TP_1 & E_{12} \\
E_{20} & E_{21} & TP_2
\end{bmatrix}$$

**a. Accuracy:**

$$\text{Accuracy} = \frac{TP_0 + TP_1 + TP_2}{\sum_{i,j} CM_{ij}}$$

**b. Precision (per class):**

$$\text{Precision}_k = \frac{TP_k}{\sum_{i} CM_{ik}}$$

**c. Recall (per class):**

$$\text{Recall}_k = \frac{TP_k}{\sum_{j} CM_{kj}}$$

**d. F1-Score (per class):**

$$F1_k = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

**e. Weighted Average:**

$$\text{Metric}_{\text{weighted}} = \sum_{k=1}^{C} \frac{n_k}{N} \cdot \text{Metric}_k$$

di mana:
- $n_k$ = jumlah sampel kelas $k$.

## Implementation

## **1. Struktur Arsitektur Model **

### **`model_builder.py`**

### **1.1 Struktur Arsitektur Model**

```python
class ModelBuilder:
    def __init__(self, config: Config):
        self.config = config
```

Kelas ModelBuilder berfungsi sebagai factory untuk membangun model Deep Learning. Konstruktor menerima  objek config yang mengandung seluruh konfigurasi sistem dan hyperparameter untuk memastikan konsistensi konfigurasi seluruh komponen

### **1.2 Pemilihan dan Konfigurasi Base Model**

```python
def build_base_model(self) -> tf.keras.Model:
    if self.config.BASE_MODEL == 'resnet50':
        base_model = keras.applications.ResNet50(
            input_shape=(*self.config.IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
```

Method `build_base_model()` memuat model pre-trained dari Keras Applications.
`input_shape` menentukan dimensi input gambar seperti tinggi, lebar, dan channel RGB. `include_top` di-set `false` untuk menghilangkan fully-connected layer asli yang memungkinkan custom head.

### **1.3 Fine-Tuning**

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(*self.config.IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.35
)
base_model.trainable = False
```

MobileNetV2 menggunakan parameter `alpha = 0.35` untuk mengurangi jumlah parameter dan menghasilkan model yang lebih ringan dan cepat.

### **1.4 Preprocessing Layer**

```python
if self.config.BASE_MODEL == 'resnet50':
    x = keras.applications.resnet50.preprocess_input(inputs)
elif self.config.BASE_MODEL == 'mobilenetv2':
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
```

Preprocessing disesuaikan dengan arsitektur base model. Karena menggunakan MobileNetV2, maka normalisasi ke range [-1,1] menggunakan layer `Rescaling`.

### **1.5 Feature Extraction dan Pooling**

```python
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
```

`training = False` memastikan batch normalization layers menggunakan statistik fixed dan `GlobalAveragePooling2d` mengkonversi feature maps menjadi vektor 1D dengan averaging spatial dimensions untuk mengurangi parameter dan mencegah overfitting. 

### **1.6 Classification Head**

```python
x = layers.Dense(100, activation='relu')(x)
x = layers.Dropout(0.2)(x)
```

### **1.7 Output Layer**

```python
outputs = layers.Dense(
    len(self.config.CLASS_NAMES),
    activation='softmax'
)(x)
```

Menggunakan jumlah neurons sesuai jumlah kelas, aktivasi softmax untuk probabilitas multi-class, dan output berupa distribusi probabilitas dengan sum  = 1.0

### **1.8 Kompilasi Model**

```python
def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            clipnorm=1.0
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
```

`complie_model` digunakan untuk mengkonfigurasi proses pembelajaran model dengan Adam Optimizer sebagai optimalisasinya dan Categorical Crossentropy sebagai Loss Function untuk klasifikasi multi-class, serta accuracy sebagai metrik evaluasi.


## **2. Evaluasi Model **

### **`evaluator.py`**

### **2.1 Struktur Model Evaluator**

```python
class ModelEvaluator:
    def __init__(self, model: tf.keras.Model, config):
        self.model = model
        self.config = config
```

Kelas `ModelEvaluator` menerima trained model dan config object untuk melakukan berbagai jenis evaluasi.

### **2.2 Evaluasi Komprehensif**

```python
def evaluate(self, test_generator) -> dict:
    predictions = self.model.predict(test_generator)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = test_generator.labels
```

proses evaluasi dilakukan dengan cara mengenerate prediksi untuk seluruh test set lalu mengonversi probabilitas ke class labels dengan `argmax`. Terakhir, diekstrak true labels dari generator.

### **2.3 Perhitungan Metrik**

```python
metrics = {
    'accuracy': accuracy_score(true_labels, pred_labels),
    'precision': precision_score(true_labels, pred_labels, average='weighted'),
    'recall': recall_score(true_labels, pred_labels, average='weighted'),
    'f1': f1_score(true_labels, pred_labels, average='weighted')
}
```

### **2.4 Classification Report**

```python
print(classification_report(
    true_labels,
    pred_labels,
    target_names=self.config.CLASS_NAMES
))
```

Report menampilkan precision, recall, F1-score untuk setiap class secara detail.

### **2.5 Confusion Matrix**

```python
def confusion_matrix(self, test_generator) -> np.ndarray:
    cm = confusion_matrix(true_labels, pred_labels)
```

Confusion matrix menunjukkan distribusi prediksi benar dan salah antar class.

### **2.6 Prediksi Single Image**

```python
def predict_single(self, image_path: str) -> dict:
        """Predict on a single image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.config.IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img)  
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            confidence = np.max(prediction)
            class_idx = np.argmax(prediction)
            class_name = self.config.CLASS_NAMES[class_idx]

            result = {
                'class': class_name,
                'confidence': float(confidence),
                'probabilities': {
                    self.config.CLASS_NAMES[i]: float(prediction[0][i])
                    for i in range(len(self.config.CLASS_NAMES))
                }
            }

            return result
```

`predict_single` digunakan untuk melakukan inferensi pada satu citra jalan dengan cara memuat dan memproses gambar agar sesuai dengan format input MobileNetV2, melakukan prediksi menggunakan model CNN yang telah dilatih, lalu mengembalikan hasil klasifikasi berupa class kerusakan jalan, confidence score, dan probabilitas 

## **3. Training **

### **`trainer.py`**

### **3.1 Struktur trainer**

```python
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.history = None
```

Kelas `Trainer` mengenkapsulasi seluruh proses training dengan menyimpan config, model, dan training history.

### **3.2 Setup Callbacks**

```python
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
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate if validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
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
```

`ModelCheckpoint` menyimpan model terbaik berdasarkan validation accuracy. Hanya model dengan  performa terbaik yang disimpan untuk mencegah overfitting. `EarlyStopping` menghentikan training jika validation los tidak membaik setelah 15 epochs. Parameter `restore_best_weights = True` mengembalikan bobot  terbaik saat training dihentikan. `ReduceLRONPlateau` mengurangi learning rate sebesar 30% jika validation loss plateau selama 7 epochs. Strategi ini membantu model menemukan minimum yang lebih baik dengan learning rate lebih kecil. `TwnsorBoard` adalah logging untuk visualisasi training di tensorBoard.

### **3.3 Main Training Loop**

```python
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
            class_weight=self.config.CLASS_WEIGHTS,  # Apply class weights
            verbose=1
        )

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return self.history
```

Training dimulai dengan build model menggunakan ModelBuilder lalu model di compile dengan optimizer dan loss function. Setelah itu model summary di print untuk  verifikasi arsitektur.

### **3.4 Model Saving**

```python
def save_model(self, path: str = './models/trained_model.keras') -> None:
        """Save trained model in .keras format (Keras 3 compatible)"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure path ends with .keras
        if not path.endswith('.keras'):
            path = path + '.keras'

        self.model.save(path)
        print(f"Model saved to {path}")
```

Model disimpan dalam format `.keras` yang menyimpan arsitektur model, bobot trained, optimizer state, dan compile configuration.

### **3.5 Confusion Matrix**

```python
def confusion_matrix(self, test_generator) -> np.ndarray:
    cm = confusion_matrix(true_labels, pred_labels)
```

Confusion matrix menunjukkan distribusi prediksi benar dan salah antar class.

## **4. Package Initialization **

### **`init.py`**

```python
"""Models package"""
from .model_builder import ModelBuilder
from .trainer import Trainer
from .evaluator import ModelEvaluator

__all__ = ['ModelBuilder', 'Trainer', 'ModelEvaluator']
```

File ini membuat directory `models/` menjadi Python package dengan membaca class dari submodules Ke package level dan mendefinisikan public API yang akan di-export saat `from models import *`.

## **5. Konfigurasi Sistem **

### **`confiig.py`**

```python
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
```

File ini berfungsi untuk menyimpan seluruh parameter sistem. `DATA_DIR` menentukan lokasi root directory dan `MODEL_DIR` menentukan lokasi penyimpanan model trained dan checkpoint. Dimensi input ditetapkan 224x224 px yang merupakan standar MobileNetV2. `BATCH_SIZE` sebesar 32 dipilih untuk stabilitas komputasi gradien efisiensi memori, serta regularisasi efek. `EPOCHS` sebesar 80 memberikan waktu cukup untuk model konvergen. `LEARNING_RATE` sebesar 0.001 sebagai default optimal untuk Adam Optimizer. 

Weight assignment digunakan untuk mengatasi dataset yang tidak imbang menggunakan strategi pengaturan weigths sebagai berikut:
- medium = 1.0
- normal = 1.5
- severe = 2.5

## **6. Helpers **

### **`6.1 gps_utils.py`**

### **a. Konversi GPS ke DMS**

```python
@staticmethod
def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal
```

### **b. Ekstrak koordinat GPS dari metadata foto**

```python
@staticmethod
def get_coordinates(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
```

### **c. Cek apakah foto memiliki metadata EXIF**

```python
exif_data = img._getexif()
if not exif_data:
    return None
```

### **d. Looping semua EXIF tags untuk mencari tag "GPSInfro" yang berisi data lokasi**

```python
for tag, value in exif_data.items():
    decoded = ExifTags.TAGS.get(tag, tag)
    if decoded == "GPSInfo":
        gps_info = value
        break
```

### **e. Ekstrak komponen GPS**

```python
lat_ref = gps_info.get(1)   # 'N' atau 'S'
lat_dms = gps_info.get(2)   # (degrees, minutes, seconds)
lon_ref = gps_info.get(3)   # 'E' atau 'W'
lon_dms = gps_info.get(4)   # (degrees, minutes, seconds)
```

### **f. Konfersi ke decimal**

```python
lat = GPSHelper.get_decimal_from_dms(lat_dms, lat_ref)
lon = GPSHelper.get_decimal_from_dms(lon_dms, lon_ref)
return lat, lon
```

### **`6.2 map_generator.py`**

### **a. Konversi foto menjadi Base64 untuk embed di HTML**

```python
@staticmethod
def encode_image(image_path, max_size=(300, 300)):
    img = Image.open(image_path)
    img.thumbnail(max_size)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=70)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"
```

### **b. Persistent storage untuk riwayat titik-titik di peta**

```python
[
  {
    "lat": -6.2084722,
    "lon": 106.8456789,
    "class": "severe",
    "conf": 0.8701,
    "img_name": "20251201_152632.jpg",
    "full_path": "/path/to/image.jpg"
  }
]
```

### **c. Load history dan filter duplikat**

```python
history_points = MapGenerator.load_history()
existing_files = {item['img_name'] for item in history_points}

new_points = []
for item in new_predictions:
    img_name = Path(img_path).name
    if img_name in existing_files: continue  # Skip duplikat
```


### **d. Ekstrak koordinat GPS**

```python
coords = GPSHelper.get_coordinates(img_path)
if coords:
    lat, lon = coords
    new_points.append({
        'lat': lat, 'lon': lon,
        'class': item['class'],
        'conf': float(item['confidence']),
        'img_name': img_name,
        'full_path': str(img_path)
    })
```

### **e. Setup peta dan center point**

```python
avg_lat = sum(p['lat'] for p in total_points) / len(total_points)
avg_lon = sum(p['lon'] for p in total_points) / len(total_points)
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)
```

### **f. setup layer groups**

```python
layer_severe = folium.FeatureGroup(name='Severe (Parah)', show=True)
layer_medium = folium.FeatureGroup(name='Medium (Sedang)', show=True)
layer_normal = folium.FeatureGroup(name='Normal (Aman)', show=False)

```
### **g. Generate pins**

```python
for point in total_points:
    # Tentukan warna & icon berdasarkan class
    if cls == 'severe':
        color, icon = 'red', 'exclamation-sign'
    elif cls == 'medium':
        color, icon = 'orange', 'warning-sign'
    else:
        color, icon = 'green', 'ok-sign'
```

### **h. Popup HTML**
```python
popup_html = f"""
<div style="font-family: Arial; width: 300px;">
    <h4>Status: {point['class'].upper()}</h4>
    <p>Akurasi: <b>{point['conf']:.2%}</b></p>
    <img src="{img_src}" style="width:100%;">
    <p style="font-size:10px;">{point['img_name']}</p>
</div>
"""
```

### **i. Tambah ke peta**
```python
popup_html = f"""
<div style="font-family: Arial; width: 300px;">
    <h4>Status: {point['class'].upper()}</h4>
    <p>Akurasi: <b>{point['conf']:.2%}</b></p>
    <img src="{img_src}" style="width:100%;">
    <p style="font-size:10px;">{point['img_name']}</p>
</div>
"""
```

### **j. Finalisasi**
```html
popup_html = f"""
<div style="font-family: Arial; width: 300px;">
    <h4>Status: {point['class'].upper()}</h4>
    <p>Akurasi: <b>{point['conf']:.2%}</b></p>
    <img src="{img_src}" style="width:100%;">
    <p style="font-size:10px;">{point['img_name']}</p>
</div>
"""
```

### **`6.3 visualizer.py`**

### **a. Generate  2 grafik accuracy dan loss**

```python
@staticmethod
def plot_training_history(history: History, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
```

### **b. Grafik accuracy**

```python
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)
```

### **c. Grafik Loss**

```python
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
```


### **d. Visualisasi  confusion matrix dengan heatmap**

```python
@staticmethod
def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str = None):
    sns.heatmap(
        cm,
        annot=True,      # Show angka di kotak
        fmt='d',         # Format integer
        cmap='Blues',    # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
```

### **e. Tanpilkan grid gambar dengan  prediksi model**

```python
@staticmethod
def show_predictions(images: list, predictions: list, true_labels: list = None, 
                    class_names: list = None, cols: int = 4):
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
```

### **f. Layout grid**

```python
for idx, (image, prediction) in enumerate(zip(images, predictions)):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col]
```

### **g. Image dan info**
```python
ax.imshow(image)
pred_class = prediction.get('class', 'Unknown')
confidence = prediction.get('confidence', 0)
title = f"Pred: {pred_class}\nConfidence: {confidence:.2%}"
```


## Demo

## **Training Result**

![Demo Sistem](../training_result.jpg)

Berdasarkan hasil training, model sudah sangat baik dan siap untuk digunakan. Dapat dilihat bahwa pada class medium, model berhasil memprediksi 10 foto berlabel "medium', 0 salah di "normal", dan 2 salah di "severe". Kemudian, pada class normal model berhasil memprediksi 11 foto berlabel "normal', 0 salah di "medium", dan 0 salah di "severe". Terakhir, pada class severe model berhasil memprediksi 10 foto berlabel "severe', 0 salah di "medium", dan 0 salah di "normal".

![Demo Sistem](../training_result2.jpg)

Berdasarkan hasil training tersebut, model sangat cepat belajar dan mencapai akurasi hampir sempurna dengan Epoch 0-5 yang naik dengan cepat dari 78%  menuju 97%, serta Epoch 5-23 yang stabil. Selain itu, model juga semakin yakin dan semakin sedikit melakukan kesalah di data training.

## **Prediction Demo**
```bash

/home/kevin/.virtualenvs/HoleHunter/bin/python /mnt/c/Users/kevin/Documents/PKA/Tubes/HoleHunter/train.py 
2025-12-14 17:54:13.112652: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
======================================================================
HOLE HUNTER - POTHOLE DETECTION MODEL TRAINING
======================================================================

📋 Configuration:
  - Base Model: mobilenetv2
  - Image Size: (224, 224)
  - Batch Size: 32
  - Epochs: 80
  - Learning Rate: 0.001
  - Classes: ['medium', 'normal', 'severe']

✅ Data directory found: ./data/processed/

📊 Creating data generators...
Found 1617 images belonging to 3 classes.
Found 30 images belonging to 3 classes.
Found 33 images belonging to 3 classes.

🚀 Starting training...
============================================================
Starting Training
============================================================
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1765706057.714498   43510 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9709 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3060, pci bus id: 0000:09:00.0, compute capability: 8.6
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling (Rescaling)           │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2,257,984 │
│ (Functional)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 100)            │       128,100 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 100)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 3)              │           303 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,386,387 (9.10 MB)
 Trainable params: 128,403 (501.57 KB)
 Non-trainable params: 2,257,984 (8.61 MB)
None
Epoch 1/80
2025-12-14 17:54:23.725777: I external/local_xla/xla/service/service.cc:163] XLA service 0x74c760003dc0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2025-12-14 17:54:23.725805: I external/local_xla/xla/service/service.cc:171]   StreamExecutor device (0): NVIDIA GeForce RTX 3060, Compute Capability 8.6
2025-12-14 17:54:23.829332: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2025-12-14 17:54:24.444317: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91700
2025-12-14 17:54:24.593579: I external/local_xla/xla/service/gpu/autotuning/dot_search_space.cc:208] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs? Working around this by using the full hints set instead.
2025-12-14 17:54:25.423936: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_4797', 68 bytes spill stores, 68 bytes spill loads

2025-12-14 17:54:29.810322: E external/local_xla/xla/service/slow_operation_alarm.cc:73] Trying algorithm eng4{} for conv (f32[32,16,112,112]{3,2,1,0}, u8[0]{0}) custom-call(f32[32,32,112,112]{3,2,1,0}, f32[16,32,1,1]{3,2,1,0}), window={size=1x1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[]} is taking a while...
2025-12-14 17:54:29.811280: E external/local_xla/xla/service/slow_operation_alarm.cc:140] The operation took 1.987374848s
Trying algorithm eng4{} for conv (f32[32,16,112,112]{3,2,1,0}, u8[0]{0}) custom-call(f32[32,32,112,112]{3,2,1,0}, f32[16,32,1,1]{3,2,1,0}), window={size=1x1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[]} is taking a while...
2025-12-14 17:54:36.275228: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2025-12-14 17:54:36.478443: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2025-12-14 17:54:36.673430: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
I0000 00:00:1765706078.610660   43639 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
31/51 ━━━━━━━━━━━━━━━━━━━━ 8s 426ms/step - accuracy: 0.6189 - loss: 1.26832025-12-14 17:55:01.215839: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2025-12-14 17:55:01.416509: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 684ms/step - accuracy: 0.6738 - loss: 1.06762025-12-14 17:55:22.972019: E external/local_xla/xla/service/slow_operation_alarm.cc:73] Trying algorithm eng3{k11=2} for conv (f32[30,576,14,14]{3,2,1,0}, u8[0]{0}) custom-call(f32[30,576,14,14]{3,2,1,0}, f32[576,1,3,3]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, feature_group_count=576, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[]} is taking a while...
2025-12-14 17:55:23.140562: E external/local_xla/xla/service/slow_operation_alarm.cc:140] The operation took 1.168628014s
Trying algorithm eng3{k11=2} for conv (f32[30,576,14,14]{3,2,1,0}, u8[0]{0}) custom-call(f32[30,576,14,14]{3,2,1,0}, f32[576,1,3,3]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, feature_group_count=576, custom_call_target="__cudnn$convForward", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[]} is taking a while...
2025-12-14 17:55:24.390987: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2025-12-14 17:55:24.585059: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.

Epoch 1: val_accuracy improved from None to 0.70000, saving model to ./models/best_model.keras
51/51 ━━━━━━━━━━━━━━━━━━━━ 69s 984ms/step - accuracy: 0.7854 - loss: 0.6750 - val_accuracy: 0.7000 - val_loss: 0.6014 - learning_rate: 0.0010
Epoch 2/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 316ms/step - accuracy: 0.9046 - loss: 0.2971
Epoch 2: val_accuracy improved from 0.70000 to 0.76667, saving model to ./models/best_model.keras
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 356ms/step - accuracy: 0.9165 - loss: 0.2724 - val_accuracy: 0.7667 - val_loss: 0.6087 - learning_rate: 0.0010
Epoch 3/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 267ms/step - accuracy: 0.9398 - loss: 0.2066
Epoch 3: val_accuracy improved from 0.76667 to 0.80000, saving model to ./models/best_model.keras
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 307ms/step - accuracy: 0.9425 - loss: 0.2025 - val_accuracy: 0.8000 - val_loss: 0.5854 - learning_rate: 0.0010
Epoch 4/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 312ms/step - accuracy: 0.9572 - loss: 0.1572
Epoch 4: val_accuracy did not improve from 0.80000
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 344ms/step - accuracy: 0.9518 - loss: 0.1710 - val_accuracy: 0.7667 - val_loss: 0.5567 - learning_rate: 0.0010
Epoch 5/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 262ms/step - accuracy: 0.9610 - loss: 0.1514
Epoch 5: val_accuracy did not improve from 0.80000
51/51 ━━━━━━━━━━━━━━━━━━━━ 15s 294ms/step - accuracy: 0.9685 - loss: 0.1337 - val_accuracy: 0.8000 - val_loss: 0.6061 - learning_rate: 0.0010
Epoch 6/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 317ms/step - accuracy: 0.9741 - loss: 0.0999
Epoch 6: val_accuracy did not improve from 0.80000
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 350ms/step - accuracy: 0.9697 - loss: 0.1167 - val_accuracy: 0.8000 - val_loss: 0.6312 - learning_rate: 0.0010
Epoch 7/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 265ms/step - accuracy: 0.9765 - loss: 0.1042
Epoch 7: val_accuracy did not improve from 0.80000
51/51 ━━━━━━━━━━━━━━━━━━━━ 15s 300ms/step - accuracy: 0.9716 - loss: 0.1103 - val_accuracy: 0.8000 - val_loss: 0.5190 - learning_rate: 0.0010
Epoch 8/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 312ms/step - accuracy: 0.9722 - loss: 0.0958
Epoch 8: val_accuracy improved from 0.80000 to 0.83333, saving model to ./models/best_model.keras
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 351ms/step - accuracy: 0.9722 - loss: 0.1027 - val_accuracy: 0.8333 - val_loss: 0.7345 - learning_rate: 0.0010
Epoch 9/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 263ms/step - accuracy: 0.9760 - loss: 0.0852
Epoch 9: val_accuracy did not improve from 0.83333
51/51 ━━━━━━━━━━━━━━━━━━━━ 15s 297ms/step - accuracy: 0.9790 - loss: 0.0840 - val_accuracy: 0.8333 - val_loss: 0.4868 - learning_rate: 0.0010
Epoch 10/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 315ms/step - accuracy: 0.9730 - loss: 0.0989
Epoch 10: val_accuracy did not improve from 0.83333
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 349ms/step - accuracy: 0.9697 - loss: 0.1117 - val_accuracy: 0.8333 - val_loss: 0.5691 - learning_rate: 0.0010
Epoch 11/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 264ms/step - accuracy: 0.9773 - loss: 0.0670
Epoch 11: val_accuracy did not improve from 0.83333
51/51 ━━━━━━━━━━━━━━━━━━━━ 15s 296ms/step - accuracy: 0.9827 - loss: 0.0615 - val_accuracy: 0.8333 - val_loss: 0.6550 - learning_rate: 0.0010
Epoch 12/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 311ms/step - accuracy: 0.9814 - loss: 0.0746
Epoch 12: val_accuracy did not improve from 0.83333
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 344ms/step - accuracy: 0.9771 - loss: 0.0932 - val_accuracy: 0.7333 - val_loss: 0.8134 - learning_rate: 0.0010
Epoch 13/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 267ms/step - accuracy: 0.9674 - loss: 0.1224
Epoch 13: val_accuracy improved from 0.83333 to 0.86667, saving model to ./models/best_model.keras
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 305ms/step - accuracy: 0.9759 - loss: 0.0951 - val_accuracy: 0.8667 - val_loss: 0.6594 - learning_rate: 0.0010
Epoch 14/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 318ms/step - accuracy: 0.9871 - loss: 0.0479
Epoch 14: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 348ms/step - accuracy: 0.9876 - loss: 0.0466 - val_accuracy: 0.7667 - val_loss: 0.8296 - learning_rate: 0.0010
Epoch 15/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 276ms/step - accuracy: 0.9676 - loss: 0.0991
Epoch 15: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 309ms/step - accuracy: 0.9740 - loss: 0.0884 - val_accuracy: 0.8000 - val_loss: 0.7700 - learning_rate: 0.0010
Epoch 16/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 308ms/step - accuracy: 0.9868 - loss: 0.0519
Epoch 16: val_accuracy did not improve from 0.86667

Epoch 16: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 341ms/step - accuracy: 0.9845 - loss: 0.0622 - val_accuracy: 0.7333 - val_loss: 1.0265 - learning_rate: 0.0010
Epoch 17/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 276ms/step - accuracy: 0.9852 - loss: 0.0610
Epoch 17: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 307ms/step - accuracy: 0.9882 - loss: 0.0485 - val_accuracy: 0.7667 - val_loss: 0.8536 - learning_rate: 3.0000e-04
Epoch 18/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 309ms/step - accuracy: 0.9923 - loss: 0.0322
Epoch 18: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 342ms/step - accuracy: 0.9932 - loss: 0.0316 - val_accuracy: 0.7667 - val_loss: 0.8572 - learning_rate: 3.0000e-04
Epoch 19/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 272ms/step - accuracy: 0.9872 - loss: 0.0466
Epoch 19: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 304ms/step - accuracy: 0.9907 - loss: 0.0364 - val_accuracy: 0.7333 - val_loss: 0.9424 - learning_rate: 3.0000e-04
Epoch 20/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 310ms/step - accuracy: 0.9910 - loss: 0.0341
Epoch 20: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 343ms/step - accuracy: 0.9920 - loss: 0.0371 - val_accuracy: 0.7333 - val_loss: 0.9506 - learning_rate: 3.0000e-04
Epoch 21/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 270ms/step - accuracy: 0.9913 - loss: 0.0454
Epoch 21: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 301ms/step - accuracy: 0.9889 - loss: 0.0498 - val_accuracy: 0.7667 - val_loss: 0.8907 - learning_rate: 3.0000e-04
Epoch 22/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 271ms/step - accuracy: 0.9926 - loss: 0.0337
Epoch 22: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 342ms/step - accuracy: 0.9920 - loss: 0.0360 - val_accuracy: 0.7667 - val_loss: 0.7952 - learning_rate: 3.0000e-04
Epoch 23/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 282ms/step - accuracy: 0.9960 - loss: 0.0266
Epoch 23: val_accuracy did not improve from 0.86667

Epoch 23: ReduceLROnPlateau reducing learning rate to 9.000000427477062e-05.
51/51 ━━━━━━━━━━━━━━━━━━━━ 16s 314ms/step - accuracy: 0.9938 - loss: 0.0323 - val_accuracy: 0.7333 - val_loss: 0.8806 - learning_rate: 3.0000e-04
Epoch 24/80
51/51 ━━━━━━━━━━━━━━━━━━━━ 0s 269ms/step - accuracy: 0.9918 - loss: 0.0278
Epoch 24: val_accuracy did not improve from 0.86667
51/51 ━━━━━━━━━━━━━━━━━━━━ 17s 339ms/step - accuracy: 0.9938 - loss: 0.0252 - val_accuracy: 0.7667 - val_loss: 0.8437 - learning_rate: 9.0000e-05
Epoch 24: early stopping
Restoring model weights from the end of the best epoch: 9.
============================================================
Training Complete!
============================================================
✅ Model saved to models/trained_model.keras

🔍 Evaluating model...
============================================================
Evaluating Model
============================================================
1/2 ━━━━━━━━━━━━━━━━━━━━ 4s 4s/step2025-12-14 18:02:05.979695: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2025-12-14 18:02:06.167900: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2/2 ━━━━━━━━━━━━━━━━━━━━ 13s 9s/step

Classification Report:
              precision    recall  f1-score   support

      medium       1.00      0.83      0.91        12
      normal       1.00      1.00      1.00        11
      severe       0.83      1.00      0.91        10

    accuracy                           0.94        33
   macro avg       0.94      0.94      0.94        33
weighted avg       0.95      0.94      0.94        33


Metrics:
  accuracy: 0.9394
  precision: 0.9495
  recall: 0.9394
  f1: 0.9394
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step 

Confusion Matrix:
[[10  0  2]
 [ 0 11  0]
 [ 0  0 10]]

📈 Generating visualizations...
✅ Training history saved to models/training_history.png
✅ Confusion matrix saved to models/confusion_matrix.png

======================================================================
✅ TRAINING COMPLETE!
======================================================================

📁 Results saved to: ./models/
   - trained_model.keras: Final trained model
   - best_model.keras: Best checkpoint during training
   - training_history.png: Accuracy and loss curves
   - confusion_matrix.png: Model predictions analysis

🎯 Model Performance:
   - Accuracy: 0.9394
   - Precision: 0.9495
   - Recall: 0.9394
   - F1: 0.9394

Process finished with exit code 0
```

Sistem mengambil semua foto dari folder `bppselatanpredik/` kemudian dengan model AI yang telah dilatih sebelumnya akan membaca foto satu per satu. Model akan menebak tingkat kerusakan jalan dan mengklasifikasikannya ke tiga kelas yaitu medium, normal, atau severe. Setelah itu, sistem akan mengambil foto dengan metadata dan menandainya di peta interaktif dengan warna pin  berbeda-beda sesuai kelasnya.

## **Map Demo**

![Demo Sistem](../map_demo(1).gif)

## Summary

Sistem Mapping Jalan Berlubang di Kalimantan menggunakan metode CNN dapat disimpulkan bahwa sistem ini mampu mengklasifikasikan tingkat keparahan jalan berlubang menjadi tiga kelas yaitu medium, normal, dan severe. Sistem juga mampu untuk menandai letak lubang dari foto yang diinput ke peta interaktif agar mempermudah pengguna untuk menemukannya. Secara Keseluruhan, sistem yang dikembangkan menunjukkan bahwa metode CNN dengan model MobileNetV2 efektif untuk mengklasifikasikan jenis jalan berlubang di Kalimantan dan dapat dikembangkan lebih lanjut sebagai pendukung pemeliharaan infrastruktur jalanan.

## References

**Sumartha, N. N. C., Wijaya, I. G. P. S., & Bimantoro, F. (2024)**. Klasifikasi Citra Lubang pada Permukaan Jalan Beraspal dengan Metode Convolutional Neural Networks (CNN). *Journal of Computer Science and Informatics Engineering (J-Cosine), 8*(1).

**Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018)**. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).

**Xiang, Q., Wang, X., Li, R., Zhang, G., Lai, J., & Hu, Q. (2019, October)**. Fruit image classification based on Mobilenetv2 with transfer learning technique. In *Proceedings of the 3rd international conference on computer science and application engineering* (pp. 1-7).
