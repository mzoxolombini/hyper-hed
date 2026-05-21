# ILS-HED: Enhancing Holistically-Nested Edge Detection with Iterative Local Search Hyper-heuristics

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **ILS-HED**, a hyper-heuristic framework that enhances Holistically-Nested Edge Detection (HED) by automatically selecting optimal classical edge detectors (Canny, Sobel, Laplacian, Gabor) and their hyperparameters using Iterative Local Search.

## 📋 Overview

ILS-HED addresses a practical need: enhancing already-deployed HED systems without retraining the backbone. The framework:

- **Searches 5,184 configurations** of classical detectors and their parameters
- **Uses Iterative Local Search (ILS)** as a hyper-heuristic for optimal selection
- **Trains learnable fusion weights** to combine HED side outputs with classical features
- **Transfers across domains** without retraining (BSDS500 → NYU Depth, PASCAL Context)

### Key Results

| Dataset | Metric | HED Baseline | ILS-HED | Improvement | p-value |
|---------|--------|--------------|---------|-------------|---------|
| BSDS500 | ODS | 0.782 | **0.791** | +0.009 | <0.001 |
| DRIVE | F1 | 0.7812 | **0.7912** | +0.0100 | <0.001 |
| STARE | F1 | 0.7620 | **0.7715** | +0.0095 | <0.001 |
| DeepCrack | F1 | 0.6850 | **0.7120** | +0.0270 | <0.001 |
| Stone331 | F1 | 0.6720 | **0.6980** | +0.0260 | <0.001 |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.3 (for GPU acceleration)
- 4+ GB GPU memory (for BSDS500 training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/mzoxolombini/hyper-hed.git
cd hyper-hed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .```

## 🏃 Usage

### Running ILS-HED

```bash
# Run on BSDS500 with default config
python -m ils_hed.train --config configs/default.yaml

# Run on BSDS500 with dataset-specific config
python -m ils_hed.train --config configs/bsds500.yaml

# Run on DRIVE
python -m ils_hed.train --config configs/drive.yaml

# Run on STARE
python -m ils_hed.train --config configs/stare.yaml

# Run on DeepCrack
python -m ils_hed.train --config configs/deepcrack.yaml

# Run on Stone331
python -m ils_hed.train --config configs/stone331.yaml

# Run on SDNet2018
python -m ils_hed.train --config configs/sdnet2018.yaml

# Transfer test (BSDS500 → NYU Depth, PASCAL Context)
python -m ils_hed.train --config configs/transfer.yaml

# Ablation study
python -m ils_hed.train --config configs/ablation.yaml
```

### Downloading Datasets

```bash
python scripts/download_datasets.py --data_dir ./data
```

### Analysis Scripts

```bash
# Visualize results
python scripts/visualize_results.py

# Plot precision-recall curves
python scripts/plot_pr_curves.py

# Plot detector selection frequency (Table 9)
python scripts/plot_detector_frequency.py

# Statistical significance tests
python scripts/statistical_tests.py

# Extract sample images and ground truth
python scripts/extract_image_ground_truth.py
```

## 📁 Repository Structure

```
hyper-hed/
├── ils_hed/                # Python package
│   ├── __init__.py
│   ├── core/
│   │   ├── detectors.py    # Classical edge detectors + HeuristicType enum
│   │   ├── fusion.py       # Learnable fusion module
│   │   ├── ils_search.py   # ILS hyper-heuristic
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── hed_model.py    # HED network (VGG-based)
│   ├── data/
│   │   └── datasets.py     # Dataset loaders
│   ├── train.py            # Main entry point
│   └── utils.py            # Utility functions
├── configs/                # YAML configuration files
│   ├── default.yaml
│   ├── bsds500.yaml
│   ├── drive.yaml
│   ├── stare.yaml
│   ├── deepcrack.yaml
│   ├── stone331.yaml
│   ├── sdnet2018.yaml
│   ├── transfer.yaml
│   └── ablation.yaml
├── scripts/                # Analysis and utility scripts
│   ├── download_datasets.py
│   ├── visualize_results.py
│   ├── plot_pr_curves.py
│   ├── plot_detector_frequency.py
│   ├── statistical_tests.py
│   └── extract_image_ground_truth.py
├── results/
├── logs/
├── checkpoints/
├── data/
├── requirements.txt
└── setup.py
```

## 📄 Citation

If you use ILS-HED in your research, please cite:

```bibtex
@article{ils-hed2024,
  title={Hyper-heuristic Enhancement of HED},
  author={Mbini, Mzoxolo},
  journal={Expert Systems with Applications},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License.
