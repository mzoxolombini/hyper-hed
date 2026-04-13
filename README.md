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
git clone https://github.com/mzoxolombini/ils-hed.git
cd ils-hed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .