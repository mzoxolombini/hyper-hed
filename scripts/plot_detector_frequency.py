#!/usr/bin/env python3
"""
Plot detector selection frequency (Table 9 from the paper).
Shows how often each classical detector was selected by ILS across datasets.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Data from Table 9 in the paper
DETECTOR_FREQUENCY = {
    'BSDS500': {
        'Canny': 5/5,  # 5/5 runs
        'Sobel': 4/5,
        'Gabor': 3/5,
        'Laplacian': 2/5,
    },
    'DRIVE': {
        'Canny': 5/5,
        'Sobel': 5/5,
        'Gabor': 2/5,
        'Laplacian': 3/5,
    },
    'DeepCrack': {
        'Canny': 5/5,
        'Sobel': 4/5,
        'Gabor': 5/5,
        'Laplacian': 1/5,
    },
    'SDNET2018': {
        'Canny': 5/5,
        'Sobel': 5/5,
        'Gabor': 4/5,
        'Laplacian': 2/5,
    },
}

# Average weights from Table 9
AVERAGE_WEIGHTS = {
    'Canny': 0.28,
    'Sobel': 0.22,
    'Gabor': 0.18,
    'Laplacian': 0.12,
}


def plot_detector_frequency_by_dataset(output_path: Path):
    """Plot detector selection frequency across datasets (bar chart)."""
    datasets = list(DETECTOR_FREQUENCY.keys())
    detectors = ['Canny', 'Sobel', 'Laplacian', 'Gabor']
    
    # Prepare data
    data = np.zeros((len(datasets), len(detectors)))
    for i, dataset in enumerate(datasets):
        for j, detector in enumerate(detectors):
            data[i, j] = DETECTOR_FREQUENCY[dataset].get(detector, 0) * 100  # Convert to percentage
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for j, detector in enumerate(detectors):
        bars = ax.bar(x + j * width, data[:, j], width, label=detector, color=colors[j])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Figure: Detector Selection Frequency Across Datasets\n(Table 9 from paper)', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Detector frequency plot saved to {output_path}")


def plot_average_weights(output_path: Path):
    """Plot average fusion weights for each detector (pie chart)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    detectors = list(AVERAGE_WEIGHTS.keys())
    weights = list(AVERAGE_WEIGHTS.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    wedges, texts, autotexts = ax.pie(weights, labels=detectors, autopct='%1.0f%%',
                                       colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    
    ax.set_title('Average Fusion Weights by Detector Type', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Average weights pie chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot detector selection frequency')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Detector Selection Frequency Plotter")
    print("=" * 60)
    
    # Generate plots
    plot_detector_frequency_by_dataset(output_dir / 'detector_frequency.png')
    plot_average_weights(output_dir / 'average_weights.png')
    
    print("\n" + "=" * 60)
    print(f"All plots saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()