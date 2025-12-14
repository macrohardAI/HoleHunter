import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Config

class ModelBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build_base_model(self) -> tf.keras.Model:
        """Load pre-trained base model"""
        
        if self.config.BASE_MODEL == 'resnet50':
            base_model = keras.applications.ResNet50(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
            # ResNet biasanya butuh fine-tuning di layer agak dalam karena arsitekturnya beda
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
                
        elif self.config.BASE_MODEL == 'mobilenetv2':
            # REVISI: Tambahkan alpha=0.35 agar mirip Teachable Machine (ringan)
            # Jika ingin akurasi lebih tinggi tapi agak lambat, ganti alpha=1.0
            base_model = keras.applications.MobileNetV2(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet',
                alpha=0.35 # Sesuaikan dengan file .h5 kamu sebelumnya
            )
            # Untuk MobileNet/Teachable Machine, biasanya base model dibekukan total dulu
            base_model.trainable = False 
            
        else:
            raise ValueError(f"Unknown base model: {self.config.BASE_MODEL}")

        return base_model

    def build_full_model(self) -> tf.keras.Model:
        base_model = self.build_base_model()
        inputs = keras.Input(shape=(*self.config.IMG_SIZE, 3))

        # --- REVISI: DYNAMIC PREPROCESSING ---
        # ResNet dan MobileNet butuh preprocessing beda.
        if self.config.BASE_MODEL == 'resnet50':
            # ResNet pakai fungsi preprocessing bawaan (caffe style)
            x = keras.applications.resnet50.preprocess_input(inputs)
        elif self.config.BASE_MODEL == 'mobilenetv2':
            # MobileNetV2 (Teachable Machine) butuh range -1 s.d 1
            # Kita pakai layer Rescaling agar aman
            x = layers.Rescaling(1./127.5, offset=-1)(inputs)
        else:
            x = inputs

        # Base model inference
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        # --- PILIHAN ARSITEKTUR HEAD ---
        
        # OPSI A: Style "Teachable Machine" (Ringan & Cepat)
        # Gunakan ini jika ingin hasil mirip file .h5 kamu
        # ------------------------------------------------
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(0.2)(x) 
        # ------------------------------------------------

        # OPSI B: Style "Deep/Complex" (Kode aslimu)
        # Gunakan ini jika data banyak & butuh akurasi maksimal
        # (Uncomment jika ingin pakai ini)
        # ------------------------------------------------
        # x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.4)(x)
        # ------------------------------------------------

        # Output layer (Pastikan config.CLASS_NAMES isinya 3 item)
        outputs = layers.Dense(
            len(self.config.CLASS_NAMES),
            activation='softmax'
        )(x)

        model = keras.Model(inputs, outputs)
        return model

    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE,
                # clipnorm bagus untuk kestabilan training
                clipnorm=1.0 
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy'] # Metrics lain opsional, boleh ditambahkan
        )
        return model