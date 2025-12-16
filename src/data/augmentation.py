"""
Data Augmentation Script
Generates multiple augmented versions of images to expand dataset size.
Transforms 1 image into 10+ variants using rotation, flip, zoom, and color adjustments.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import random


class DataAugmentor:
    """Data augmentation class for generating multiple variants of images"""

    def __init__(self, seed=42):
        """
        Initialize augmentor

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def rotate(image, angle_range=30):
        """Randomly rotate image"""
        angle = random.randint(-angle_range, angle_range)
        return image.rotate(angle, fillcolor='white', expand=False)

    @staticmethod
    def horizontal_flip(image):
        """Horizontal flip"""
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    @staticmethod
    def vertical_flip(image):
        """Vertical flip"""
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    @staticmethod
    def zoom(image, zoom_range=0.2):
        """Random zoom (crop and resize)"""
        width, height = image.size
        zoom_factor = 1 + random.uniform(-zoom_range, zoom_range)

        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        # Center crop
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def shift(image, shift_range=0.15):
        """Random shift (translation)"""
        width, height = image.size

        shift_x = int(width * random.uniform(-shift_range, shift_range))
        shift_y = int(height * random.uniform(-shift_range, shift_range))

        # Create new image with white background
        new_image = Image.new('RGB', (width, height), 'white')
        new_image.paste(image, (shift_x, shift_y))

        return new_image

    @staticmethod
    def brightness(image, brightness_range=0.3):
        """Random brightness adjustment"""
        factor = random.uniform(1 - brightness_range, 1 + brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def contrast(image, contrast_range=0.3):
        """Random contrast adjustment"""
        factor = random.uniform(1 - contrast_range, 1 + contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def saturation(image, saturation_range=0.3):
        """Random saturation adjustment"""
        factor = random.uniform(1 - saturation_range, 1 + saturation_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def hue_shift(image, hue_range=30):
        """Random hue shift (color rotation)"""
        # Convert to HSV
        hsv_image = image.convert('HSV')
        pixels = hsv_image.load()
        width, height = hsv_image.size

        hue_shift = random.randint(-hue_range, hue_range)

        for y in range(height):
            for x in range(width):
                h, s, v = pixels[x, y]
                # Shift hue
                h = (h + hue_shift) % 256
                pixels[x, y] = (h, s, v)

        return hsv_image.convert('RGB')

    @staticmethod
    def gaussian_blur(image, radius_range=(0.1, 0.5)):
        """Random gaussian blur"""
        radius = random.uniform(*radius_range)
        return image.filter(__import__('PIL.ImageFilter', fromlist=['GaussianBlur']).GaussianBlur(radius))

    @staticmethod
    def shear(image, shear_range=0.2):
        """Random shear transformation"""
        width, height = image.size
        shear_factor = random.uniform(-shear_range, shear_range)

        # Shear transformation coefficients
        shear_matrix = (
            1, shear_factor, -shear_factor * height / 2,
            0, 1, 0
        )

        return image.transform((width, height), Image.Transform.AFFINE, shear_matrix, Image.Resampling.BILINEAR)

    def augment_image(self, image, num_variants=10):
        """
        Generate multiple augmented variants of an image

        Args:
            image: PIL Image object
            num_variants: Number of augmented variants to create

        Returns:
            List of augmented PIL Image objects
        """
        augmented_images = [image.copy()]  # Include original

        augmentation_functions = [
            lambda img: self.rotate(img),
            lambda img: self.horizontal_flip(img),
            lambda img: self.zoom(img),
            lambda img: self.shift(img),
            lambda img: self.brightness(img),
            lambda img: self.contrast(img),
            lambda img: self.saturation(img),
            lambda img: self.shear(img),
            lambda img: self.gaussian_blur(img),
        ]

        # Generate variants by combining augmentations
        for i in range(num_variants):
            img = image.copy()

            # Apply 1-3 random augmentations
            num_transforms = random.randint(1, 3)
            selected_transforms = random.sample(augmentation_functions, num_transforms)

            for transform in selected_transforms:
                try:
                    img = transform(img)
                except Exception as e:
                    print(f"Warning: Augmentation failed: {e}")

            augmented_images.append(img)

        return augmented_images


def augment_dataset(
        input_dir,
        output_dir=None,
        num_variants=10,
        augment_in_place=False,
        image_extensions=None
):
    """
    Augment all images in a directory structure

    Args:
        input_dir: Input directory (can be data/raw or data/processed)
        output_dir: Output directory (if None and augment_in_place=False, creates augmented/ subdirectory)
        num_variants: Number of augmented variants per image
        augment_in_place: If True, add augmented images to same directory. If False, create new structure.
        image_extensions: List of image extensions to process (default: jpg, jpeg, png, bmp, heic)
    """

    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.heic'}

    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Determine output directory
    if output_dir is None:
        if augment_in_place:
            output_path = input_path
        else:
            output_path = input_path.parent / f"{input_path.name}_augmented"
    else:
        output_path = Path(output_dir)

    augmentor = DataAugmentor()
    total_originals = 0
    total_generated = 0

    print("=" * 70)
    print("Data Augmentation")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Variants per image: {num_variants}")
    print("=" * 70)

    # Find all image files
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        print("-" * 70)

        # Get images in this class
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not images:
            print(f"  No images found")
            continue

        print(f"  Found {len(images)} images")

        # Create output directory
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        class_total_generated = 0

        for i, image_path in enumerate(images):
            try:
                # Open and convert image
                img = Image.open(image_path).convert('RGB')

                # Generate augmented variants
                augmented = augmentor.augment_image(img, num_variants=num_variants)

                # Save original
                original_name = image_path.stem
                original_ext = image_path.suffix

                if augment_in_place:
                    # Save original back if not already there
                    original_out = output_class_dir / image_path.name
                    if not original_out.exists():
                        img.save(original_out, quality=95)
                else:
                    original_out = output_class_dir / image_path.name
                    img.save(original_out, quality=95)

                # Save augmented variants (skip first one which is original)
                for variant_idx, aug_img in enumerate(augmented[1:], 1):
                    variant_name = f"{original_name}_aug_{variant_idx:02d}{original_ext}"
                    variant_path = output_class_dir / variant_name
                    aug_img.save(variant_path, quality=95)
                    class_total_generated += 1

                total_originals += 1
                total_generated += num_variants

                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{len(images)} images...")

            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")

        print(f"  Generated: {num_variants} variants × {len(images)} = {class_total_generated} augmented images")

    print("\n" + "=" * 70)
    print("Augmentation Complete!")
    print("=" * 70)
    print(f"Original images:   {total_originals}")
    print(f"Augmented variants: {total_generated}")
    print(f"Total images:      {total_originals + total_generated}")
    print(f"Expansion factor:  {1 + (total_generated / total_originals if total_originals > 0 else 0):.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Augment dataset to expand number of images')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory (data/raw or data/processed)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: input_dir_augmented)')
    parser.add_argument('--variants', type=int, default=10,
                        help='Number of augmented variants per image (default: 10)')
    parser.add_argument('--in-place', action='store_true',
                        help='Add augmented images to same directory instead of creating new one')

    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_variants=args.variants,
        augment_in_place=args.in_place
    )