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

## Methods

<!-- Insert methods -->
Metode yang digunakan untuk membangun sitem ini adalah Convulutional Neural Network yang merupakan salah satu arsitektur dari Deep Learning. CNN yang digunakan menggunakan <nama model> dan akan dilatih untuk mendeteksi dan mengklasifikasikan jalan berlubang menjadi tiga kelas yaitu Severe, Medium, dan Normal.

### Convulutional Operations 
```bash
Output(i,j) = Σ Σ Input(i+m, j+n) × Kernel(m,n) + bias
```

### Residual Block 
```bash
F(x) = H(x) - x
H(x) = F(x) + x
```

### Activation Function
```bash
ReLU(x) = max(0, x)
```

### Output Layer
```bash
Softmax(zi) = e^zi / Σ e^zj
```

### Loss Function 
```bash
Loss = -1/N Σ [yi × log(ŷi) + (1-yi) × log(1-ŷi)]
```

### Optimizer
```bash
mt = β1×mt-1 + (1-β1)×gt        ← momentum
vt = β2×vt-1 + (1-β2)×gt²       ← variance
θt = θt-1 - α × mt/√(vt + ε)    ← weight update
```
α = learning rate (0.001)

### Evaluation Metrics
```bash
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Normalization
```bash
x_normalized = (x - μ) / σ
```


## Implementation

<!-- Insert implementation -->

## Demo

<!-- Insert demo -->

## Summary

<!-- Insert summary -->

## References

**Sumartha, N. N. C., Wijaya, I. G. P. S., & Bimantoro, F. (2024)**. Klasifikasi Citra Lubang pada Permukaan Jalan Beraspal dengan Metode Convolutional Neural Networks (CNN). *Journal of Computer Science and Informatics Engineering (J-Cosine), 8*(1).
