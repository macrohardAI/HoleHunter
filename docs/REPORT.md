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

## Demo

## **Training Result**

![Demo Sistem](../assets/demo.gif)

Berdasarkan hasil training, model sudah sangat baik dan siap untuk digunakan. Dapat dilihat bahwa pada class medium, model berhasil memprediksi 10 foto berlabel "medium', 0 salah di "normal", dan 2 salah di "severe". Kemudian, pada class normal model berhasil memprediksi 11 foto berlabel "normal', 0 salah di "medium", dan 0 salah di "severe". Terakhir, pada class severe model berhasil memprediksi 10 foto berlabel "severe', 0 salah di "medium", dan 0 salah di "normal".


## **Prediction Demo**
![Demo Sistem](../assets/demo.gif)
![Demo Sistem](../assets/demo.gif)
![Demo Sistem](../assets/demo.gif)

Sistem mengambil semua foto dari folder `bppselatanpredik/` kemudian dengan model AI yang telah dilatih sebelumnya akan membaca foto satu per satu. Model akan menebak tingkat kerusakan jalan dan mengklasifikasikannya ke tiga kelas yaitu medium, normal, atau severe. Setelah itu, sistem akan mengambil foto dengan metadata dan menandainya di peta interaktif dengan warna pin  berbeda-beda sesuai kelasnya.

## **Map Demo**

![Demo Sistem](../assets/demo.gif)

## Summary

Sistem Mapping Jalan Berlubang di Kalimantan menggunakan metode CNN dapat disimpulkan bahwa sistem ini mampu mengklasifikasikan tingkat keparahan jalan berlubang menjadi tiga kelas yaitu medium, normal, dan severe. Sistem juga mampu untuk menandai letak lubang dari foto yang diinput ke peta interaktif agar mempermudah pengguna untuk menemukannya. Secara Keseluruhan, sistem yang dikembangkan menunjukkan bahwa metode CNN dengan model MobileNetV2 efektif untuk mengklasifikasikan jenis jalan berlubang di Kalimantan dan dapat dikembangkan lebih lanjut sebagai pendukung pemeliharaan infrastruktur jalanan.

## References

**Sumartha, N. N. C., Wijaya, I. G. P. S., & Bimantoro, F. (2024)**. Klasifikasi Citra Lubang pada Permukaan Jalan Beraspal dengan Metode Convolutional Neural Networks (CNN). *Journal of Computer Science and Informatics Engineering (J-Cosine), 8*(1).

**Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018)**. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).

**Xiang, Q., Wang, X., Li, R., Zhang, G., Lai, J., & Hu, Q. (2019, October)**. Fruit image classification based on Mobilenetv2 with transfer learning technique. In *Proceedings of the 3rd international conference on computer science and application engineering* (pp. 1-7).
