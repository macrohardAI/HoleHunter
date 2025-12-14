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

Jalan berlubang dapat mengganggu kenyamanan pengendara di lalu lintas. Selain itu, jalan berlubang juga berpotensi membahayakan pengendara terutama para pengemudi motor di. Kalimantan banyak menggunakan transportasi darat untuk bepergian antar kota yang pastinya memerlukan jalan yang layak untuk dilewati. Karena adanya masalah tersebut, mendorong penulis untuk menciptakan suatu sistem yang mampu mengklasifikasikan jalan berlubang melalui foto yang memiliki metadata dengan menerapkan metode Convolutional Neural Network (CNN). 

**Keywords:** Deep Learning, Transfer Learning, MobileNetV2, Road Damage Detection, Image Classification

## Methods

Klasifikasi jalan berlubang menggunakan teknik klasifikasi gambar dengan lebih dari satu kelas menggunakan gambar input berukuran 224 px x 224 px RGB. Sistem  ini menggunakan Transfer Learning dengan MobileNetV2 sebagai base model. HoleHunter bertujuan untuk memprediksi tingkat keparahan kerusakan jalan dan  mengklasifikasikannya ke dalam salah satu dari tiga kelas yaitu:
- Medium (Rusak ringan)
- Normal
- Severe (Rusak berat)

## **2.1 MobileNetV2**
Kami menggunakan transfer learning dengan MobileNetV2 sebagai base model karena efisiensinya untuk deployments  mobile dan akurasi yang  lebih tinggi dari model lain.

### **1. Input (224×224×3)**

### **2. Rescalling Normalization**
Input gambar dinormalisasi ke range [-1,1] sesuai dengan kebutuhan MobleNetV2.

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

<!-- Insert implementation -->

## Demo

<!-- Insert demo -->

## Summary

<!-- Insert summary -->

## References

**Sumartha, N. N. C., Wijaya, I. G. P. S., & Bimantoro, F. (2024)**. Klasifikasi Citra Lubang pada Permukaan Jalan Beraspal dengan Metode Convolutional Neural Networks (CNN). *Journal of Computer Science and Informatics Engineering (J-Cosine), 8*(1).

**Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018)**. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).

**Xiang, Q., Wang, X., Li, R., Zhang, G., Lai, J., & Hu, Q. (2019, October)**. Fruit image classification based on Mobilenetv2 with transfer learning technique. In *Proceedings of the 3rd international conference on computer science and application engineering* (pp. 1-7).
