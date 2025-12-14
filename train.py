"""
Main training script - orchestrates the entire training pipeline
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from data.preprocessor import DataPreprocessor
from models.trainer import Trainer
from models.evaluator import ModelEvaluator
from helpers.visualizer import Visualizer


def main():
    print("=" * 70)
    print("HOLE HUNTER - POTHOLE DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Configuration
    config = Config()
    print(f"\n📋 Configuration:")
    print(f"  - Base Model: {config.BASE_MODEL}")
    print(f"  - Image Size: {config.IMG_SIZE}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print(f"  - Classes: {config.CLASS_NAMES}")
    
    # Check if data exists
    train_dir = Path(config.DATA_DIR) / 'processed' / 'train'
    val_dir = Path(config.DATA_DIR) / 'processed' / 'validation'
    test_dir = Path(config.DATA_DIR) / 'processed' / 'test'
    
    if not train_dir.exists() or not val_dir.exists():
        print("\n❌ Error: Processed data not found!")
        print(f"   Please run: python data_preparation.py first")
        print(f"   Expected directories:")
        print(f"     - {train_dir}")
        print(f"     - {val_dir}")
        return
    
    print(f"\n✅ Data directory found: {config.DATA_DIR}/processed/")
    
    # Create data generators
    print("\n📊 Creating data generators...")
    preprocessor = DataPreprocessor(config)
    train_datagen, val_test_datagen = preprocessor.create_data_generators()
    
    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        str(val_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        str(test_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Train model
    print("\n🚀 Starting training...")
    trainer = Trainer(config)
    history = trainer.train(train_generator, val_generator)
    
    model_path = Path(config.MODEL_DIR) / 'trained_model.keras'
    trainer.save_model(str(model_path))

    # Evaluate model
    print("\n🔍 Evaluating model...")
    evaluator = ModelEvaluator(trainer.model, config)
    metrics = evaluator.evaluate(test_generator)

    # Generate confusion matrix
    cm = evaluator.confusion_matrix(test_generator)

    # Visualize results
    print("\n📈 Generating visualizations...")
    viz = Visualizer()
    viz.plot_training_history(history, 'models/training_history.png')
    viz.plot_confusion_matrix(cm, config.CLASS_NAMES, 'models/confusion_matrix.png')

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {config.MODEL_DIR}/")
    print(f"   - trained_model.keras: Final trained model")
    print(f"   - best_model.keras: Best checkpoint during training")
    print(f"   - training_history.png: Accuracy and loss curves")
    print(f"   - confusion_matrix.png: Model predictions analysis")

    print("\n🎯 Model Performance:")
    for metric_name, value in metrics.items():
        print(f"   - {metric_name.capitalize()}: {value:.4f}")


if __name__ == '__main__':
    main()
