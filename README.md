# ILS-HED: Enhancing Holistically-Nested Edge Detection with Iterative Local Search Hyper-heuristics

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **ILS-HED**, a hyper-heuristic framework that enhances Holistically-Nested Edge Detection (HED) by automatically selecting optimal classical edge detectors (Canny, Sobel, Laplacian, Gabor) and fusing them with HED side outputs via learnable weights — without retraining the HED backbone.

> **Paper:** *Hyper-heuristic Enhancement of HED* — submitted to Expert Systems With Applications (ESWA).  
> See [`Hyper_heuristic_Enhancement_of_HED___ESWA.pdf`](Hyper_heuristic_Enhancement_of_HED___ESWA.pdf) for the full manuscript.

---

## 📋 Overview

ILS-HED addresses a practical need: enhancing already-deployed HED systems without retraining the backbone. The framework:

- **Searches 5,184 configurations** of classical detectors and their parameters
- **Uses Iterative Local Search (ILS)** as a hyper-heuristic for optimal detector selection
- **Trains learnable fusion weights** to combine HED side outputs (S1–S5) with selected classical features
- **Transfers across domains** without retraining (BSDS500 → DRIVE, STARE, DeepCrack, Stone331)

### Key Results

| Dataset   | Metric | HED Baseline | ILS-HED   | Improvement | p-value |
|-----------|--------|--------------|-----------|-------------|---------|
| BSDS500   | ODS    | 0.782        | **0.791** | +0.009      | <0.001  |
| DRIVE     | F1     | 0.7812       | **0.7912**| +0.0100     | <0.001  |
| STARE     | F1     | 0.7620       | **0.7715**| +0.0095     | <0.001  |
| DeepCrack | F1     | 0.6850       | **0.7120**| +0.0270     | <0.001  |
| Stone331  | F1     | 0.6720       | **0.6980**| +0.0260     | <0.001  |

---

## 🗂️ Repository Structure

```
hyper-hed/
├── ils_hed/                          # Python package
│   ├── __init__.py
│   ├── core/
│   │   ├── detectors.py              # Classical edge detectors (Canny, Sobel, Laplacian, Gabor)
│   │   ├── fusion.py                 # Learnable fusion module
│   │   ├── ils_search.py             # ILS hyper-heuristic search
│   │   ├── metrics.py                # ODS, F1, PR-curve, bootstrap CI
│   │   └── hed_model.py              # HED network (VGG-16 backbone, side outputs S1–S5)
│   ├── data/
│   │   └── datasets.py               # Dataset loaders (BSDS500, DRIVE, STARE, DeepCrack, Stone331, SDNet2018)
│   ├── utils.py                      # Utility functions
│   └── train.py                      # Main entry point
├── configs/
│   ├── default.yaml                  # Default ILS hyperparameters
│   ├── bsds500.yaml                  # BSDS500 experiment config
│   ├── drive.yaml                    # DRIVE experiment config
│   ├── stare.yaml                    # STARE experiment config
│   ├── deepcrack.yaml                # DeepCrack experiment config
│   ├── stone331.yaml                 # Stone331 experiment config
│   ├── sdnet2018.yaml                # SDNet2018 experiment config
│   ├── transfer.yaml                 # Cross-domain transfer config
│   └── ablation.yaml                 # Ablation study config
├── scripts/
│   ├── download_datasets.py          # Dataset download helper
│   ├── extract_image_ground_truth.py # Ground-truth extraction
│   ├── visualize_results.py          # Result visualisation
│   ├── plot_pr_curves.py             # PR-curve plots
│   ├── plot_detector_frequency.py    # Detector selection frequency plots
│   └── statistical_tests.py         # Wilcoxon signed-rank + bootstrap CI tests
├── data/                             # Dataset root (populated by download script)
├── results/                          # Output predictions and metrics
├── logs/                             # Training logs
├── checkpoints/                      # Saved model weights
├── requirements.txt
├── setup.py
└── Hyper_heuristic_Enhancement_of_HED___ESWA.pdf
```

---

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
pip install -e .
```

---

## 📦 Datasets

Download datasets using the provided script:

```bash
python scripts/download_datasets.py --data_dir ./data --datasets BSDS500 DeepCrack SDNET2018
```

Datasets that require manual registration (DRIVE, STARE) will print instructions. Place them under `./data/` following the structure below:

```
data/
├── BSDS500/
│   ├── images/{train,val,test}/
│   └── ground_truth/{train,val,test}/
├── DRIVE/
│   ├── training/{images,1st_manual,mask}/
│   └── test/{images,1st_manual,mask}/
├── STARE/                    # .ppm images in root
├── DeepCrack/
│   ├── image/
│   └── ground_truth/
├── Stone331/
├── Stone331_mask/
└── SDNET2018/
    ├── Decks/{Cracked,Uncracked}/
    ├── Pavements/{Cracked,Uncracked}/
    └── Walls/{Cracked,Uncracked}/
```

### Pretrained HED weights

Download the pretrained HED weights (VGG-16, trained on BSDS500) and place them at the repo root:

```bash
wget https://vcl.ucsd.edu/hed/hed_pretrained_bsds.pth
```

---

## 🧪 Usage

### Run ILS search on BSDS500

```bash
python -m ils_hed.train --config configs/bsds500.yaml
```

### Run on a specific dataset

```bash
python -m ils_hed.train --config configs/drive.yaml
python -m ils_hed.train --config configs/deepcrack.yaml
python -m ils_hed.train --config configs/stare.yaml
```

### Run all datasets

```bash
python -m ils_hed.train --config configs/default.yaml --run_all
```

### Cross-domain transfer

```bash
python -m ils_hed.train --config configs/transfer.yaml
```

### Ablation study

```bash
python -m ils_hed.train --config configs/ablation.yaml
```

---

## 📊 Visualisation & Analysis

```bash
# Visualise edge detection results
python scripts/visualize_results.py --results_dir ./results --dataset BSDS500

# Plot precision-recall curves
python scripts/plot_pr_curves.py --results_dir ./results

# Plot detector selection frequency (Figure 5 in paper)
python scripts/plot_detector_frequency.py --results_dir ./results

# Statistical significance tests (Table 6 in paper)
python scripts/statistical_tests.py --results_dir ./results
```

---

## ⚙️ Configuration

All hyperparameters are controlled via YAML files in `configs/`. Key parameters (from `configs/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ils.max_iterations` | 50 | Maximum ILS iterations |
| `ils.patience` | 10 | Early stopping patience (P) |
| `ils.fusion_epochs` | 5 | Epochs to train fusion weights per evaluation |
| `ils.learning_rate` | 0.0001 | Fusion module learning rate |
| `fusion.fusion_type` | `learnable` | `learnable` or `average` |
| `postprocessing.thinning_method` | `morphological` | `morphological` or `nms` |

---

## 🏗️ Framework Architecture

```
Input Image
    │
    ▼
┌─────────────────────┐
│   HED Network       │  ← LOCKED (pretrained, not fine-tuned)
│  (VGG-16 backbone)  │
│  S1, S2, S3, S4, S5 │  ← 5 side outputs (locked heuristics)
└─────────────────────┘
    │
    ├── S1–S5 (locked)
    │
    ├── Canny   ┐
    ├── Sobel   │ ← Classical detectors selected by ILS (variable heuristics)
    ├── Laplace │
    └── Gabor   ┘
         │
         ▼
┌─────────────────────┐
│  Learnable Fusion   │  ← Softmax-weighted combination
│  w₁S₁+…+w₅S₅+wᶜCᶜ  │
└─────────────────────┘
         │
         ▼
   Enhanced Edge Map
```

The ILS hyper-heuristic uses three perturbation operators (add / remove / swap) to explore the space of classical detector subsets, evaluating each candidate via HED-weighted binary cross-entropy.

---

## 📈 Ablation Study

Key ablation results on BSDS500 (Table 8 in paper):

| Configuration | ODS | OIS | AP |
|---|---|---|---|
| (a) HED S1–S5 (original) | 0.782 | 0.804 | 0.833 |
| (c) HED + Best Single (Canny, learned) | 0.785 | 0.807 | 0.838 |
| (e) HED + ILS Selection (average) | 0.788 | 0.810 | 0.842 |
| (f) HED + ILS Params Only (learned) | 0.789 | 0.811 | 0.844 |
| **(g) HED + ILS Selection (learned) — Ours** | **0.791** | **0.813** | **0.847** |
| (h) HED + All Traditional (learned) | 0.784 | 0.806 | 0.835 |
| (i) HED + Greedy Forward Selection | 0.789 | 0.811 | 0.844 |

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{mbini2024ilshed,
  title     = {Hyper-heuristic Enhancement of Holistically-Nested Edge Detection},
  author    = {Mbini, Mzoxolo},
  journal   = {Expert Systems With Applications},
  year      = {2024}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Original HED implementation: [s9xie/hed](https://github.com/s9xie/hed)
- BSDS500 benchmark: [UC Berkeley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
