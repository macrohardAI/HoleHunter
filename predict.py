"""
Prediction script - inference on new images
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from models.evaluator import ModelEvaluator
from helpers.visualizer import Visualizer
from helpers.map_generator import MapGenerator


def predict_image(model_path: str, image_path: str, config: Config = None):
    """Predict on a single image"""
    if config is None:
        config = Config()

    if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
        model_path = model_path + '.keras'

    print(f"\n Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except ValueError as e:
        if "Unknown layer" in str(e):
            print(f"!!!  Error loading model. Trying to rebuild from architecture...")
            # Fallback: rebuild model if loading fails
            from models.model_builder import ModelBuilder
            builder = ModelBuilder(config)
            model = builder.build_full_model()
            model = builder.compile_model(model)
            print("!!!  Model rebuilt. Note: weights may not be loaded.")
        else:
            raise

    print(f" Predicting on: {image_path}")
    evaluator = ModelEvaluator(model, config)

    result = evaluator.predict_single(image_path)

    if result:
        result['image_path'] = str(image_path)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Image: {Path(image_path).name}")
        print(f"Classification: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities for all classes:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            bar_length = int(prob * 30)  # Visual bar representation
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {class_name:20s}: [{bar}] {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 50)

        print("\n  Updating Map...")
        MapGenerator.generate_map([result], 'laporan_peta_kerusakan.html')
        print("=" * 50)

        # Display image
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            viz = Visualizer()
            viz.show_predictions([np.array(img)], [result])
        except Exception as e:
            print(f"!!!  Could not display image: {e}")

    return result


def predict_batch(model_path: str, image_dir: str, config: Config = None):
    """Predict on multiple images in a directory"""
    if config is None:
        config = Config()

    if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
        model_path = model_path + '.keras'

    print(f"\n Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except ValueError as e:
        if "Unknown layer" in str(e):
            print(f"!!!  Error loading model. Trying to rebuild from architecture...")
            from models.model_builder import ModelBuilder
            builder = ModelBuilder(config)
            model = builder.build_full_model()
            model = builder.compile_model(model)
            print("!!!  Model rebuilt. Note: weights may not be loaded.")
        else:
            raise

    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.heic'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        print(f"X No images found in {image_dir}")
        return

    print(f" Found {len(images)} images to predict")

    evaluator = ModelEvaluator(model, config)
    results = []
    images_array = []

    for img_path in images:
        try:
            result = evaluator.predict_single(str(img_path))
            if result:
                result['image_path'] = str(img_path)

                results.append(result)

                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images_array.append(np.array(img))
                print(f"   {img_path.name}: {result['class']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"  ⚠️  {img_path.name}: Error - {str(e)}")
            continue

        
    if results:
        print("\n" + "=" * 50)
        MapGenerator.generate_map(results, 'laporan_peta_kerusakan.html')
        print("=" * 50)

        viz = Visualizer()
        viz.show_predictions(images_array, results)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict on images using trained model')
    parser.add_argument('--model', default='models/trained_model.keras',
                        help='Path to trained model (.keras format)')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--batch', help='Directory of images for batch prediction')

    args = parser.parse_args()

    config = Config()

    if args.image:
        predict_image(args.model, args.image, config)
    elif args.batch:
        predict_batch(args.model, args.batch, config)
    else:
        print("Please provide either --image or --batch argument")
        print(f"Usage: python predict.py --image <image_path>")
        print(f"       python predict.py --batch <directory>")
