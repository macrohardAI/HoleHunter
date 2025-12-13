"""
Data Preparation Script
Organizes raw images into train/validation/test splits
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image


def prepare_dataset(
        raw_data_dir='data/raw',
        processed_dir='data/processed',
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        target_size=(224, 224),
        seed=42
):
    """
    Prepare dataset by splitting into train/val/test and organizing folders

    Args:
        raw_data_dir: Directory containing class folders (e.g., raw/berlubang, raw/rusak_ringan, raw/normal)
        processed_dir: Output directory for processed data
        train_split: Proportion of data for training (0.7 = 70%)
        val_split: Proportion for validation (0.15 = 15%)
        test_split: Proportion for testing (0.15 = 15%)
        target_size: Resize images to this size (width, height)
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    print("=" * 60)
    print("Data Preparation Script (Multi-Class)")
    print("=" * 60)

    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 0.001:
        raise ValueError("Splits must sum to 1.0")

    raw_path = Path(raw_data_dir)
    if not raw_path.exists():
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")

    # Get all class folders
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    classes.sort()  # Sort for consistency

    if not classes:
        raise ValueError(f"No class folders found in {raw_data_dir}")

    print(f"Found {len(classes)} classes: {', '.join(classes)}")

    splits = ['train', 'validation', 'test']

    # Create directory structure
    for split in splits:
        for cls in classes:
            path = Path(processed_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)

    # Process each class
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        print("-" * 60)

        # Get all images in this class
        raw_class_dir = Path(raw_data_dir) / cls

        if not raw_class_dir.exists():
            print(f"⚠️  Warning: {raw_class_dir} does not exist. Skipping...")
            continue

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.heic'}
        images = [
            f for f in raw_class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if len(images) == 0:
            print(f"⚠️  No images found in {raw_class_dir}")
            continue

        print(f"Found {len(images)} images")

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        n = len(images)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

        # Copy and resize images
        def copy_and_resize(image_list, split_name):
            dest_dir = Path(processed_dir) / split_name / cls
            success_count = 0
            error_count = 0

            for img_path in image_list:
                try:
                    # Open and resize image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                    # Save to destination
                    dest_path = dest_dir / img_path.name
                    img.save(dest_path, quality=95)
                    success_count += 1

                except Exception as e:
                    print(f"  ❌ Error processing {img_path.name}: {e}")
                    error_count += 1

            return success_count, error_count

        # Process each split
        train_ok, train_err = copy_and_resize(train_images, 'train')
        val_ok, val_err = copy_and_resize(val_images, 'validation')
        test_ok, test_err = copy_and_resize(test_images, 'test')

        print(f"  ✅ Train: {train_ok} processed, {train_err} errors")
        print(f"  ✅ Val:   {val_ok} processed, {val_err} errors")
        print(f"  ✅ Test:  {test_ok} processed, {test_err} errors")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)

    for split in splits:
        print(f"\n{split.upper()}:")
        for cls in classes:
            path = Path(processed_dir) / split / cls
            count = len(list(path.glob('*')))
            print(f"  {cls}: {count} images")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Verify the data splits look correct")
    print("2. Run: python train.py")
    print("=" * 60)


def verify_dataset(processed_dir='data/processed'):
    """
    Verify dataset structure and count images
    """
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    processed_path = Path(processed_dir)
    splits = sorted([d.name for d in processed_path.iterdir() if d.is_dir()])

    if not splits:
        print(f"No splits found in {processed_dir}")
        return

    total_images = 0

    for split in splits:
        print(f"\n{split.upper()}:")
        split_path = processed_path / split
        split_total = 0

        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])

        for cls in classes:
            path = split_path / cls
            if path.exists():
                count = len(list(path.glob('*')))
                print(f"  {cls}: {count} images")
                split_total += count
            else:
                print(f"  {cls}: Directory not found!")
        print(f"  Total: {split_total}")
        total_images += split_total

    print(f"\n{'=' * 60}")
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)


def create_sample_dataset(output_dir='data/raw', classes_config=None, num_samples_per_class=10):
    """
    Create a dummy dataset for testing (when you don't have real data yet)
    Creates random colored images for testing the pipeline

    Args:
        output_dir: Output directory
        classes_config: List of class names. Default: ['berlubang', 'rusak_ringan', 'normal']
        num_samples_per_class: Number of samples to create per class
    """
    if classes_config is None:
        classes_config = ['medium', 'normal', 'severe']

    print("\n" + "=" * 60)
    print("Creating Sample Dataset (for testing only)")
    print("=" * 60)

    for cls in classes_config:
        class_dir = Path(output_dir) / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            # Create random colored image
            img = Image.new('RGB', (640, 480),
                            color=(random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255)))

            # Save
            img.save(class_dir / f'sample_{cls}_{i:03d}.jpg')

        print(f"✅ Created {num_samples_per_class} sample images for '{cls}'")

    print("\n⚠️  Note: These are dummy images for testing the pipeline only!")
    print("Replace with real road condition images before actual training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--raw-dir', default='data/raw',
                        help='Directory with raw images organized by class')
    parser.add_argument('--output-dir', default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training split (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Test split (default: 0.15)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Target image size (default: 224)')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample dataset for testing')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify existing dataset')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset(args.raw_dir)

    if args.verify:
        verify_dataset(args.output_dir)
    else:
        prepare_dataset(
            raw_data_dir=args.raw_dir,
            processed_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            target_size=(args.image_size, args.image_size)
        )