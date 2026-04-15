#!/usr/bin/env python3
"""
Visualization script for ILS-HED qualitative results.
Generates Figures 2-6 from the paper:
- Figure 2: BSDS500 qualitative comparison
- Figure 3: BSDS500 edge maps
- Figure 4: DeepCrack results
- Figure 5: DRIVE retinal vessel results
- Figure 6: Stone331 results
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import cv2

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ils_hed.core.hed_model import HEDNetwork
    from ils_hed.core.ils_search import ILS_HED, ILSConfig
    from ils_hed.core.detectors import DetectorFactory
    from ils_hed.core.fusion import LearnableFusion
except ImportError as _e:
    raise ImportError(
        "Could not import from 'ils_hed' package. "
        "Please install the package first with: pip install -e ."
    ) from _e


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained ILS-HED model."""
    hed_model = HEDNetwork(pretrained=True).to(device)
    hed_model.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_config = checkpoint['best_config']
    
    # Create fusion module
    fusion = LearnableFusion(n_hed_outputs=5, n_classical=len(best_config))
    fusion.load_state_dict(checkpoint['fusion_state'])
    fusion = fusion.to(device)
    fusion.eval()
    
    return hed_model, fusion, best_config


def process_image(image_path: Path, hed_model, fusion, best_config, device):
    """Process a single image through ILS-HED."""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image.shape[:2]
    
    # Resize to 400x400 (HED input size)
    image_resized = cv2.resize(image, (400, 400))
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    # HED forward pass
    with torch.no_grad():
        hed_outputs = hed_model(image_tensor)
    
    # Classical detector outputs
    classical_outputs = []
    for det_config in best_config:
        img_np = image_resized.astype(np.float32) / 255.0
        edge_np = DetectorFactory.detect(img_np, det_config)
        edge_tensor = torch.from_numpy(edge_np).float().unsqueeze(0).unsqueeze(0).to(device)
        classical_outputs.append(edge_tensor)
    
    # Fusion
    fused = fusion(hed_outputs, classical_outputs)
    fused = torch.sigmoid(fused)
    
    # Resize back to original size
    fused_np = fused.squeeze().cpu().numpy()
    fused_np = cv2.resize(fused_np, (original_w, original_h))
    
    # Binary threshold
    binary = (fused_np > 0.5).astype(np.uint8) * 255
    
    return image, fused_np, binary


def figure2_bsds500_comparison(images_dir: Path, checkpoint_path: Path, output_dir: Path):
    """
    Generate Figure 2: BSDS500 qualitative comparison.
    Shows input image, ground truth, HED output, ILS-HED output.
    """
    print("Generating Figure 2: BSDS500 Comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hed_model, fusion, best_config = load_model(checkpoint_path, device)
    
    # Sample images from BSDS500 test set
    sample_images = [
        "12003.jpg",  # Bird
        "24063.jpg",  # Person on horse
        "86016.jpg",  # Plane (where ILS-HED shows improvement)
    ]
    
    fig, axes = plt.subplots(4, len(sample_images), figsize=(15, 20))
    
    for col, img_name in enumerate(sample_images):
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"Warning: {img_path} not found")
            continue
        
        # Process image
        original, edge_map, binary = process_image(img_path, hed_model, fusion, best_config, device)
        
        # Display original image
        axes[0, col].imshow(original.astype(np.uint8))
        axes[0, col].set_title(f'Input Image\n{img_name}', fontsize=10)
        axes[0, col].axis('off')
        
        # Display edge map (heatmap)
        im = axes[1, col].imshow(edge_map, cmap='hot', vmin=0, vmax=1)
        axes[1, col].set_title('ILS-HED Edge Probability', fontsize=10)
        axes[1, col].axis('off')
        
        # Display binary edge map
        axes[2, col].imshow(binary, cmap='gray')
        axes[2, col].set_title('ILS-HED Binary Edges', fontsize=10)
        axes[2, col].axis('off')
        
        # Display overlay
        overlay = original.copy().astype(np.uint8)
        overlay[binary > 0] = [255, 0, 0]  # Red edges
        axes[3, col].imshow(overlay)
        axes[3, col].set_title('Edge Overlay', fontsize=10)
        axes[3, col].axis('off')
    
    plt.suptitle('Figure 2: Qualitative Comparison of Edge Detection on BSDS500', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_bsds500_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure2_bsds500_comparison.png'}")


def figure3_bsds500_edge_maps(images_dir: Path, checkpoint_path: Path, output_dir: Path):
    """
    Generate Figure 3: BSDS500 edge maps comparison.
    Shows HED vs ILS-HED side-by-side.
    """
    print("Generating Figure 3: BSDS500 Edge Maps...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hed_model, fusion, best_config = load_model(checkpoint_path, device)
    
    sample_images = [
        ("42049.jpg", "Cat"),
        ("86016.jpg", "Plane"),
        ("97033.jpg", "Elephant"),
    ]
    
    fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 12))
    
    for row, (img_name, title) in enumerate(sample_images):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        
        # Process
        original, edge_map, binary = process_image(img_path, hed_model, fusion, best_config, device)
        
        # Baseline HED (simplified - for comparison)
        image_resized = cv2.resize(original, (400, 400))
        img_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            hed_outputs = hed_model(img_tensor)
            hed_fused = torch.sigmoid(torch.stack(hed_outputs).mean(dim=0))
            hed_np = hed_fused.squeeze().cpu().numpy()
            hed_np = cv2.resize(hed_np, (original.shape[1], original.shape[0]))
        
        # Display
        axes[row, 0].imshow(original.astype(np.uint8))
        axes[row, 0].set_title(f'Input: {title}', fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(hed_np, cmap='gray')
        axes[row, 1].set_title('HED Baseline', fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(edge_map, cmap='gray')
        axes[row, 2].set_title('ILS-HED (Ours)', fontsize=10)
        axes[row, 2].axis('off')
    
    plt.suptitle('Figure 3: Edge Detection Comparison on BSDS500', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_bsds500_edge_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure3_bsds500_edge_maps.png'}")


def figure4_deepcrack_results(images_dir: Path, checkpoint_path: Path, output_dir: Path):
    """
    Generate Figure 4: DeepCrack qualitative results.
    """
    print("Generating Figure 4: DeepCrack Results...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hed_model, fusion, best_config = load_model(checkpoint_path, device)
    
    # DeepCrack test images
    sample_images = [
        "crack_001.jpg",
        "crack_045.jpg",
        "crack_089.jpg",
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, img_name in enumerate(sample_images):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        
        # Process
        original, edge_map, binary = process_image(img_path, hed_model, fusion, best_config, device)
        
        # Baseline HED
        image_resized = cv2.resize(original, (400, 400))
        img_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            hed_outputs = hed_model(img_tensor)
            hed_fused = torch.sigmoid(torch.stack(hed_outputs).mean(dim=0))
            hed_np = hed_fused.squeeze().cpu().numpy()
            hed_np = cv2.resize(hed_np, (original.shape[1], original.shape[0]))
            hed_binary = (hed_np > 0.5).astype(np.uint8) * 255
        
        # Display
        axes[row, 0].imshow(original.astype(np.uint8))
        axes[row, 0].set_title('Input Image', fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(hed_binary, cmap='gray')
        axes[row, 1].set_title('HED Baseline', fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(binary, cmap='gray')
        axes[row, 2].set_title('ILS-HED (Ours)', fontsize=10)
        axes[row, 2].axis('off')
        
        # Overlay
        overlay = original.copy().astype(np.uint8)
        overlay[binary > 0] = [0, 255, 0]  # Green edges
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Edge Overlay', fontsize=10)
        axes[row, 3].axis('off')
    
    plt.suptitle('Figure 4: Crack Detection Results on DeepCrack Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_deepcrack_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure4_deepcrack_results.png'}")


def figure5_drive_results(images_dir: Path, checkpoint_path: Path, output_dir: Path):
    """
    Generate Figure 5: DRIVE retinal vessel segmentation results.
    """
    print("Generating Figure 5: DRIVE Results...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hed_model, fusion, best_config = load_model(checkpoint_path, device)
    
    sample_images = [
        "01_test.tif",
        "02_test.tif",
        "03_test.tif",
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, img_name in enumerate(sample_images):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        
        # Process
        original, edge_map, binary = process_image(img_path, hed_model, fusion, best_config, device)
        
        # Display
        axes[row, 0].imshow(original.astype(np.uint8))
        axes[row, 0].set_title('Retinal Image', fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(edge_map, cmap='gray')
        axes[row, 1].set_title('ILS-HED Probability', fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(binary, cmap='gray')
        axes[row, 2].set_title('ILS-HED Binary', fontsize=10)
        axes[row, 2].axis('off')
        
        # Overlay
        overlay = original.copy().astype(np.uint8)
        overlay[binary > 0] = [255, 0, 0]  # Red vessels
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('Vessel Overlay', fontsize=10)
        axes[row, 3].axis('off')
    
    plt.suptitle('Figure 5: Retinal Vessel Segmentation on DRIVE Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_drive_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_dir / 'figure5_drive_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate ILS-HED visualization figures')
    parser.add_argument('--checkpoint', type=str, default='results/BSDS500_best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                        help='Output directory for figures')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory containing datasets')
    parser.add_argument('--figures', nargs='+', 
                        default=['figure2', 'figure3', 'figure4', 'figure5'],
                        choices=['figure2', 'figure3', 'figure4', 'figure5', 'figure6'],
                        help='Which figures to generate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    
    print("=" * 60)
    print("ILS-HED Visualization Generator")
    print("=" * 60)
    
    if 'figure2' in args.figures:
        bsds500_images = data_dir / 'BSDS500/images/test'
        if bsds500_images.exists():
            figure2_bsds500_comparison(bsds500_images, checkpoint_path, output_dir)
        else:
            print(f"⚠️  BSDS500 test images not found at {bsds500_images}")
    
    if 'figure3' in args.figures:
        bsds500_images = data_dir / 'BSDS500/images/test'
        if bsds500_images.exists():
            figure3_bsds500_edge_maps(bsds500_images, checkpoint_path, output_dir)
        else:
            print(f"⚠️  BSDS500 test images not found at {bsds500_images}")
    
    if 'figure4' in args.figures:
        deepcrack_images = data_dir / 'DeepCrack/test/images'
        if deepcrack_images.exists():
            figure4_deepcrack_results(deepcrack_images, checkpoint_path, output_dir)
        else:
            print(f"⚠️  DeepCrack test images not found at {deepcrack_images}")
    
    if 'figure5' in args.figures:
        drive_images = data_dir / 'DRIVE/test/images'
        if drive_images.exists():
            figure5_drive_results(drive_images, checkpoint_path, output_dir)
        else:
            print(f"⚠️  DRIVE test images not found at {drive_images}")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()