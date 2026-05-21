#!/usr/bin/env python3
"""
Precision-Recall curve plotting for ILS-HED.
Generates precision-recall curves comparing HED baseline vs ILS-HED.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ils_hed.core.hed_model import HEDNetwork
from ils_hed.core.ils_search import ILS_HED, ILSConfig
from ils_hed.core.detectors import DetectorFactory
from ils_hed.core.fusion import LearnableFusion
from ils_hed.data.datasets import create_dataloaders


def evaluate_predictions(model, fusion, best_config, dataloader, device):
    """Generate all predictions for a dataset."""
    all_predictions = []
    all_targets = []
    
    model.eval()
    fusion.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            targets = batch['edge_map'].to(device)
            
            # HED outputs
            hed_outputs = model(images)
            
            # Classical detector outputs
            classical_outputs = []
            for det_config in best_config:
                img_np = images[0].cpu().numpy().transpose(1, 2, 0)
                edge_np = DetectorFactory.detect(img_np, det_config)
                edge_tensor = torch.from_numpy(edge_np).float().to(device)
                classical_outputs.append(edge_tensor.unsqueeze(0).unsqueeze(0))
            
            # Fuse
            fused = fusion(hed_outputs, classical_outputs)
            fused = torch.sigmoid(fused)
            
            all_predictions.append(fused.cpu().numpy().ravel())
            all_targets.append(targets.cpu().numpy().ravel())
    
    return np.concatenate(all_predictions), np.concatenate(all_targets)


def plot_pr_curves(hed_predictions, hed_targets, ils_predictions, ils_targets, output_path):
    """Plot precision-recall curves comparing HED and ILS-HED."""
    # Compute precision-recall curves
    hed_precision, hed_recall, _ = precision_recall_curve(hed_targets, hed_predictions)
    ils_precision, ils_recall, _ = precision_recall_curve(ils_targets, ils_predictions)
    
    # Compute AP scores
    hed_ap = average_precision_score(hed_targets, hed_predictions)
    ils_ap = average_precision_score(ils_targets, ils_predictions)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(hed_recall, hed_precision, 'b-', linewidth=2, 
            label=f'HED Baseline (AP = {hed_ap:.3f})')
    ax.plot(ils_recall, ils_precision, 'r-', linewidth=2,
            label=f'ILS-HED (AP = {ils_ap:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves on BSDS500 Test Set', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  HED AP: {hed_ap:.4f}")
    print(f"  ILS-HED AP: {ils_ap:.4f}")
    print(f"  Improvement: +{ils_ap - hed_ap:.4f}")
    
    return hed_ap, ils_ap


def main():
    parser = argparse.ArgumentParser(description='Plot precision-recall curves')
    parser.add_argument('--dataset', type=str, default='BSDS500',
                        help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default='results/BSDS500_best.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='results/figures/pr_curves.png',
                        help='Output path for figure')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Precision-Recall Curve Generator")
    print("=" * 60)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    _, _, test_loader = create_dataloaders(args.dataset, args.data_dir, batch_size=1)
    
    # Load model
    print("Loading ILS-HED model...")
    hed_model = HEDNetwork(pretrained=True).to(device)
    hed_model.eval()
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    best_config = checkpoint['best_config']
    
    fusion = LearnableFusion(n_hed_outputs=5, n_classical=len(best_config))
    fusion.load_state_dict(checkpoint['fusion_state'])
    fusion = fusion.to(device)
    fusion.eval()
    
    # Baseline HED (simple fusion)
    class BaselineHED(torch.nn.Module):
        def __init__(self, hed_model):
            super().__init__()
            self.hed_model = hed_model
        def forward(self, x):
            outputs = self.hed_model(x)
            return torch.sigmoid(torch.stack(outputs).mean(dim=0))
    
    baseline = BaselineHED(hed_model).to(device)
    baseline.eval()
    
    # Evaluate
    print("Evaluating HED baseline...")
    baseline_preds, baseline_targets = evaluate_predictions(
        hed_model, None, [], test_loader, device
    )
    
    print("Evaluating ILS-HED...")
    ils_preds, ils_targets = evaluate_predictions(
        hed_model, fusion, best_config, test_loader, device
    )
    
    # Plot
    print("Generating precision-recall curves...")
    plot_pr_curves(baseline_preds, baseline_targets, ils_preds, ils_targets, output_path)
    
    print(f"\n✅ PR curves saved to {output_path}")


if __name__ == '__main__':
    main()