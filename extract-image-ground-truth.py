#!/usr/bin/env python3
"""
Extract 1 sample image and ground truth from each dataset
Fixed version - uses actual file names from folder structure
"""

import os
import numpy as np
import cv2
from skimage import io
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
base_data_dir = r"C:\Users\mzoxo\OneDrive\Documents\hyp-data"

# Define paths using actual file names from folder structure
datasets = {
    'BSDS500': {
        'image': os.path.join(base_data_dir, 'BSDS500', 'images', 'train', '100075.jpg'),
        'gt': os.path.join(base_data_dir, 'BSDS500', 'ground_truth', 'train', '100075.mat'),
        'type': 'mat'
    },
    'DeepCrack': {
        'image': os.path.join(base_data_dir, 'DeepCrack', 'image', '6192.jpg'),
        'gt': os.path.join(base_data_dir, 'DeepCrack', 'ground_truth', '6192.bmp'),
        'type': 'image'
    },
    'Stone331': {
        'image': os.path.join(base_data_dir, 'Stone331', '738.jpg'),
        'gt': os.path.join(base_data_dir, 'Stone331_mask', '738.bmp'),
        'type': 'image'
    },
    'CrackLS315': {
        'image': os.path.join(base_data_dir, 'CrackLS315', '0001-2.jpg'),
        'gt': None,
        'type': 'none'
    },
    'CRKWH100': {
        'image': os.path.join(base_data_dir, 'CRKWH100', '1000.png'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Decks_Cracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Decks', 'Cracked', '7001-115.jpg'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Decks_NonCracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Decks', 'Non-cracked', '7001-1.jpg'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Pavements_Cracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Pavements', 'Cracked', '001-100.jpg'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Pavements_NonCracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Pavements', 'Non-cracked', '001-1.jpg'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Walls_Cracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Walls', 'Cracked', '7069-101.jpg'),
        'gt': None,
        'type': 'none'
    },
    'SDNET_Walls_NonCracked': {
        'image': os.path.join(base_data_dir, 'SDNET', 'Walls', 'Non-cracked', '7069-1.jpg'),
        'gt': None,
        'type': 'none'
    },
    'DRIVE_Training': {
        'image': os.path.join(base_data_dir, 'DRIVE', 'training', 'images', '21_training.tif'),
        'gt': os.path.join(base_data_dir, 'DRIVE', 'training', '1st_manual', '21_manual1.gif'),
        'mask': os.path.join(base_data_dir, 'DRIVE', 'training', 'mask', '21_training_mask.gif'),
        'type': 'vessel'
    },
    'DRIVE_Test': {
        'image': os.path.join(base_data_dir, 'DRIVE', 'test', 'images', '01_test.tif'),
        'gt': None,  # Test set doesn't have public ground truth in this structure
        'type': 'none'
    },
    'STARE': {
        'image': os.path.join(base_data_dir, 'STARE', 'im0001.ppm'),
        'gt': None,
        'type': 'none'
    }
}


def load_bsds500_gt(mat_path):
    """Load BSDS500 ground truth boundaries"""
    try:
        mat = loadmat(mat_path)
        boundaries = []
        for i in range(mat['groundTruth'].shape[1]):
            item = mat['groundTruth'][0, i]
            if 'Boundaries' in item.dtype.names:
                boundaries.append(item['Boundaries'][0, 0].astype(np.float32))
        if boundaries:
            return np.maximum.reduce(boundaries)
    except Exception as e:
        print(f"    Error loading BSDS500 GT: {e}")
    return None


def load_drive_gt(gt_path, mask_path=None):
    """Load DRIVE ground truth and mask - handles dimension issues"""
    try:
        # Load ground truth
        gt_img = Image.open(gt_path)
        gt = np.array(gt_img)

        # Handle different dimensions
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]  # Take first channel if RGB
        gt = (gt > 0).astype(np.float32)

        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0)

        return gt, mask
    except Exception as e:
        print(f"    Error loading DRIVE GT: {e}")
        return None, None


def load_generic_image(img_path, as_gray=False):
    """Load image safely with PIL first, fallback to skimage"""
    try:
        img = Image.open(img_path)
        if as_gray:
            img = img.convert('L')
            img = np.array(img)
        else:
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        return img
    except Exception as e:
        print(f"    Error loading image: {e}")
        return None


def load_generic_gt(gt_path):
    """Load generic ground truth image"""
    try:
        gt_img = Image.open(gt_path)
        gt = np.array(gt_img)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt = (gt > 0).astype(np.float32)
        return gt
    except Exception as e:
        print(f"    Error loading GT: {e}")
        return None


def extract_sample(dataset_name, info):
    """Extract image and ground truth for a dataset"""
    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 50}")

    # Load image
    img = None
    if os.path.exists(info['image']):
        img = load_generic_image(info['image'])
        if img is not None:
            print(f"  ✓ Image loaded: {os.path.basename(info['image'])}")
            print(f"    Shape: {img.shape}, dtype: {img.dtype}")
            print(f"    Range: [{img.min():.3f}, {img.max():.3f}]")
    else:
        print(f"  ✗ Image not found: {info['image']}")

    # Load ground truth
    gt = None
    mask = None

    if info['type'] == 'mat' and info['gt'] and os.path.exists(info['gt']):
        print(f"  Loading GT from MAT file...")
        gt = load_bsds500_gt(info['gt'])
        if gt is not None:
            print(f"  ✓ GT loaded: {os.path.basename(info['gt'])}")
            print(f"    Shape: {gt.shape}, unique values: {np.unique(gt)}")

    elif info['type'] == 'image' and info['gt'] and os.path.exists(info['gt']):
        print(f"  Loading GT from image...")
        gt = load_generic_gt(info['gt'])
        if gt is not None:
            print(f"  ✓ GT loaded: {os.path.basename(info['gt'])}")
            print(f"    Shape: {gt.shape}, positive pixels: {gt.sum():.0f}")

    elif info['type'] == 'vessel':
        if info['gt'] and os.path.exists(info['gt']):
            print(f"  Loading DRIVE GT and mask...")
            gt, mask = load_drive_gt(info['gt'], info.get('mask'))
            if gt is not None:
                print(f"  ✓ GT loaded: {os.path.basename(info['gt'])}")
                print(f"    Shape: {gt.shape}, positive pixels: {gt.sum():.0f}")
                if mask is not None:
                    print(f"  ✓ Mask loaded: {os.path.basename(info.get('mask', ''))}")
                    print(f"    Shape: {mask.shape}, valid pixels: {mask.sum():.0f}")

    else:
        print(f"  ℹ Ground Truth: Not available for this dataset")

    return img, gt, mask


def visualize_sample(dataset_name, img, gt, mask=None, save_path=None):
    """Create visualization of the sample - handles dimension issues"""
    if img is None:
        print(f"  Cannot visualize: No image loaded")
        return

    # Ensure image is in correct format for display
    if len(img.shape) == 2:
        img_display = img
        is_rgb = False
    elif img.shape[2] == 3:
        # Normalize if needed
        if img.max() <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)
        is_rgb = True
    else:
        img_display = img[:, :, 0]
        is_rgb = False

    # Determine number of subplots
    if gt is not None and mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Input Image', 'Ground Truth', 'FOV Mask']
        images = [img_display, gt, mask]
        cmaps = [None if is_rgb else 'gray', 'gray', 'gray']
    elif gt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ['Input Image', 'Ground Truth']
        images = [img_display, gt]
        cmaps = [None if is_rgb else 'gray', 'gray']
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        titles = ['Input Image']
        images = [img_display]
        cmaps = [None if is_rgb else 'gray']
        axes = [axes]  # Make it iterable

    # Display each image
    for idx, (ax, img_data, title, cmap) in enumerate(zip(axes, images, titles, cmaps)):
        # Handle 3D arrays for ground truth
        if len(img_data.shape) == 3 and img_data.shape[0] == 1:
            img_data = img_data.squeeze(0)  # Remove first dimension if it's 1
        elif len(img_data.shape) == 3 and img_data.shape[2] == 1:
            img_data = img_data[:, :, 0]

        # For binary images, ensure they're 2D
        if len(img_data.shape) == 3 and title != 'Input Image':
            img_data = img_data[:, :, 0]

        ax.imshow(img_data, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved: {save_path}")

    plt.show()
    plt.close()


def main():
    """Extract one sample from each dataset"""
    print("=" * 60)
    print("EXTRACTING ONE SAMPLE FROM EACH DATASET")
    print("=" * 60)

    samples = {}

    for dataset_name, info in datasets.items():
        img, gt, mask = extract_sample(dataset_name, info)

        if img is not None:
            samples[dataset_name] = {
                'image': img,
                'ground_truth': gt,
                'mask': mask,
                'info': info
            }

            # Create visualization
            save_path = os.path.join(os.getcwd(), f"{dataset_name}_sample.png")
            visualize_sample(dataset_name, img, gt, mask, save_path)
        else:
            print(f"  ✗ Skipping {dataset_name} - no image loaded")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total datasets processed: {len(samples)}")

    print("\n✓ Datasets with ground truth:")
    for name, data in samples.items():
        if data['ground_truth'] is not None:
            print(f"    {name}")

    print("\n○ Datasets without ground truth (visualization only):")
    for name, data in samples.items():
        if data['ground_truth'] is None:
            print(f"    {name}")

    print("\n📁 Files saved:")
    for name in samples.keys():
        print(f"    {name}_sample.png")

    return samples


if __name__ == "__main__":
    samples = main()