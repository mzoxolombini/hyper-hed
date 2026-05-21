"""
Main entry point for ILS-HED training and evaluation.
Parses arguments/config, instantiates classes, and runs the ILS loop.
"""

import torch
import numpy as np
import random
import os
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional

from .core.hed_model import HEDNetwork
from .core.detectors import HeuristicType
from .core.fusion import FusionModule, simple_average_fusion, post_process
from .core.ils_search import ILSHyperheuristic, SolutionEvaluator, FeatureExtractor, Solution
from .core.metrics import Metrics
from .data.datasets import (
    BSDS500Dataset, DeepCrackDataset, Stone331Dataset,
    CrackLS315Dataset, CRKWH100Dataset, SDNETDataset,
    DRIVEDataset, STAREDataset
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for ILS-HED system"""
    # Base directory
    base_data_dir: str = "./data"

    # Dataset paths (will be set in __post_init__)
    bsds500_root: str = None
    deepcrack_root: str = None
    stone331_root: str = None
    stone331_mask_root: str = None
    crackls315_root: str = None
    crkwh100_root: str = None
    sdnet_root: str = None
    drive_root: str = None
    stare_root: str = None

    # HED weights
    hed_weights_path: str = "./hed_pretrained_bsds.pth"
    output_dir: str = "./ils_hed_results"

    # Processing
    image_size: Tuple[int, int] = (400, 400)
    drive_size: Tuple[int, int] = (565, 584)
    stare_size: Tuple[int, int] = (700, 605)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ILS Settings
    max_iter: int = 50
    patience: int = 10
    num_restarts: int = 2
    num_epochs: int = 5

    # Solution constraints
    min_traditional: int = 1

    # Fusion
    use_learnable_fusion: bool = True

    def __post_init__(self):
        # Set dataset paths
        if self.base_data_dir:
            self.bsds500_root = os.path.join(self.base_data_dir, "BSDS500")
            self.deepcrack_root = os.path.join(self.base_data_dir, "DeepCrack")
            self.stone331_root = os.path.join(self.base_data_dir, "Stone331")
            self.stone331_mask_root = os.path.join(self.base_data_dir, "Stone331_mask")
            self.crackls315_root = os.path.join(self.base_data_dir, "CrackLS315")
            self.crkwh100_root = os.path.join(self.base_data_dir, "CRKWH100")
            self.sdnet_root = os.path.join(self.base_data_dir, "SDNET")
            self.drive_root = os.path.join(self.base_data_dir, "DRIVE")
            self.stare_root = os.path.join(self.base_data_dir, "STARE")

        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['visualizations', 'results', 'models']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    def __init__(self, config: Config):
        self.config = config
        self.hed_model = HEDNetwork(config.hed_weights_path).to(config.device)
        self.hed_model.eval()
        self.all_results = {}

    def run_experiment(self, name: str, train_dataset, test_dataset, is_vessel: bool = False):
        """Run experiment on a dataset"""
        print(f"\n{'=' * 80}")
        print(f"{name} EXPERIMENT")
        print(f"{'=' * 80}")

        # Prepare training data
        train_data = []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            if len(item) >= 2 and item[1] is not None:
                img, gt = item[0], item[1]
                if gt is not None and gt.sum() > 0:
                    train_data.append((img, gt))

        # For datasets without GT, use pseudo-GT from HED
        if len(train_data) == 0:
            print("No GT available, using pseudo-GT from HED")
            for i in range(min(20, len(train_dataset))):
                img = train_dataset[i][0]
                features = FeatureExtractor(img, self.hed_model)
                hed_maps = [features.get(h) for h in HeuristicType.get_locked()]
                pseudo_gt = simple_average_fusion(hed_maps)
                train_data.append((img, pseudo_gt))

        print(f"Training on {len(train_data)} images")

        # Run ILS optimization
        evaluator = SolutionEvaluator(self.config, self.hed_model, self.config.use_learnable_fusion)
        ils = ILSHyperheuristic(self.config, evaluator)

        start_time = time.time()
        best_solution = ils.optimize(train_data, verbose=True)
        elapsed = time.time() - start_time

        print(f"\nBest solution: {best_solution}")
        print(f"Time: {elapsed:.2f}s")

        # Evaluate on test set
        baseline_metrics = {'f1': [], 'precision': [], 'recall': []}
        enhanced_metrics = {'f1': [], 'precision': [], 'recall': []}

        if is_vessel:
            baseline_metrics['sensitivity'] = []
            baseline_metrics['specificity'] = []
            enhanced_metrics['sensitivity'] = []
            enhanced_metrics['specificity'] = []

        test_indices = []
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            if len(item) >= 2:
                gt = item[1] if len(item) > 1 else None
                if gt is not None and gt.sum() > 0:
                    test_indices.append(i)

        if len(test_indices) == 0:
            test_indices = list(range(min(20, len(test_dataset))))

        print(f"Evaluating on {len(test_indices)} images...")

        for idx in tqdm(test_indices):
            item = test_dataset[idx]
            img = item[0]
            gt = item[1] if len(item) > 1 else None
            name = item[2] if len(item) > 2 else f"img_{idx}"
            mask = item[2] if len(item) > 3 and is_vessel else None

            if gt is None:
                continue

            features = FeatureExtractor(img, self.hed_model)

            # Baseline: HED only
            hed_maps = [features.get(h) for h in HeuristicType.get_locked()]
            baseline_pred = simple_average_fusion(hed_maps)
            baseline_post = post_process(baseline_pred)

            # Enhanced: HED + selected traditional
            all_maps = [features.get(h) for h in best_solution.all_heuristics]
            if self.config.use_learnable_fusion and best_solution.fusion_weights is not None:
                enhanced_pred = np.zeros_like(all_maps[0])
                for i, w in enumerate(best_solution.fusion_weights):
                    enhanced_pred += w * all_maps[i]
                enhanced_pred = 1.0 / (1.0 + np.exp(-np.clip(2 * enhanced_pred - 1, -50, 50)))
            else:
                enhanced_pred = simple_average_fusion(all_maps)
            enhanced_post = post_process(enhanced_pred)

            # Compute metrics
            if is_vessel:
                # Find optimal thresholds
                best_f1_base, thresh_base = 0, 0.5
                best_f1_enh, thresh_enh = 0, 0.5
                for thresh in np.linspace(0.1, 0.9, 50):
                    f1_base, _, _ = Metrics.compute_f1(baseline_post, gt, thresh)
                    if f1_base > best_f1_base:
                        best_f1_base = f1_base
                        thresh_base = thresh
                    f1_enh, _, _ = Metrics.compute_f1(enhanced_post, gt, thresh)
                    if f1_enh > best_f1_enh:
                        best_f1_enh = f1_enh
                        thresh_enh = thresh

                f1_base, prec_base, rec_base = Metrics.compute_f1(baseline_post, gt, thresh_base)
                sens_base, spec_base = Metrics.compute_sensitivity_specificity(baseline_post, gt, mask, thresh_base)

                f1_enh, prec_enh, rec_enh = Metrics.compute_f1(enhanced_post, gt, thresh_enh)
                sens_enh, spec_enh = Metrics.compute_sensitivity_specificity(enhanced_post, gt, mask, thresh_enh)

                baseline_metrics['sensitivity'].append(sens_base)
                baseline_metrics['specificity'].append(spec_base)
                enhanced_metrics['sensitivity'].append(sens_enh)
                enhanced_metrics['specificity'].append(spec_enh)
            else:
                f1_base, prec_base, rec_base = Metrics.compute_f1(baseline_post, gt, 0.5)
                f1_enh, prec_enh, rec_enh = Metrics.compute_f1(enhanced_post, gt, 0.5)

            baseline_metrics['f1'].append(f1_base)
            baseline_metrics['precision'].append(prec_base)
            baseline_metrics['recall'].append(rec_base)
            enhanced_metrics['f1'].append(f1_enh)
            enhanced_metrics['precision'].append(prec_enh)
            enhanced_metrics['recall'].append(rec_enh)

        # Print results
        print(f"\n{name} Results:")
        print(f"{'Metric':<15} {'Baseline HED':<15} {'ILS-HED':<15} {'Improvement':<15}")
        print("-" * 60)

        for key in baseline_metrics.keys():
            b_mean = np.mean(baseline_metrics[key])
            e_mean = np.mean(enhanced_metrics[key])
            imp = e_mean - b_mean
            print(f"{key:<15} {b_mean:.4f}        {e_mean:.4f}        {imp:+.4f}")

        # Store results
        self.all_results[name] = {
            'best_solution': best_solution.to_dict(),
            'baseline': {k: float(np.mean(v)) for k, v in baseline_metrics.items()},
            'enhanced': {k: float(np.mean(v)) for k, v in enhanced_metrics.items()},
            'time_seconds': elapsed
        }

        # Save visualizations
        self._save_visualizations(test_dataset, test_indices[:10], best_solution, name)

        return best_solution

    def _save_visualizations(self, dataset, indices, solution: Solution, name: str):
        save_dir = os.path.join(self.config.output_dir, 'visualizations', name)
        os.makedirs(save_dir, exist_ok=True)

        for idx in indices:
            item = dataset[idx]
            img = item[0]
            gt = item[1] if len(item) > 1 else None
            img_name = item[2] if len(item) > 2 else f"img_{idx}"

            features = FeatureExtractor(img, self.hed_model)

            # Baseline
            hed_maps = [features.get(h) for h in HeuristicType.get_locked()]
            baseline_pred = simple_average_fusion(hed_maps)
            baseline_post = post_process(baseline_pred)

            # Enhanced
            all_maps = [features.get(h) for h in solution.all_heuristics]
            enhanced_pred = simple_average_fusion(all_maps)
            enhanced_post = post_process(enhanced_pred)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Show original
            if img.shape[2] == 3:
                axes[0].imshow(img)
            else:
                axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            # Show GT
            if gt is not None:
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth')
            else:
                axes[1].text(0.5, 0.5, 'No GT', ha='center', va='center')
                axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # Show baseline
            axes[2].imshow(baseline_post, cmap='gray')
            axes[2].set_title('HED Baseline')
            axes[2].axis('off')

            # Show enhanced
            axes[3].imshow(enhanced_post, cmap='gray')
            axes[3].set_title('ILS-HED (Ours)')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{img_name}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to {save_dir}")

    def run_all(self):
        """Run experiments on all datasets"""
        print("\n" + "=" * 80)
        print("ILS-HYPERHEURISTIC HED - COMPLETE EVALUATION")
        print("=" * 80)

        # BSDS500
        train_bsds = BSDS500Dataset(self.config, 'train')
        val_bsds = BSDS500Dataset(self.config, 'val')
        test_bsds = BSDS500Dataset(self.config, 'test')

        # Combine train and val
        class CombinedDataset:
            def __init__(self, ds1, ds2):
                self.ds1 = ds1
                self.ds2 = ds2

            def __len__(self): return len(self.ds1) + len(self.ds2)

            def __getitem__(self, idx):
                if idx < len(self.ds1):
                    return self.ds1[idx]
                return self.ds2[idx - len(self.ds1)]

        combined_train = CombinedDataset(train_bsds, val_bsds)
        self.run_experiment("BSDS500", combined_train, test_bsds, is_vessel=False)

        # DeepCrack
        deepcrack = DeepCrackDataset(self.config)
        if len(deepcrack) > 0:
            n_train = max(5, len(deepcrack) // 3)
            train_dc = [deepcrack[i] for i in range(n_train)]
            test_dc = deepcrack
            self.run_experiment("DeepCrack", train_dc, test_dc, is_vessel=False)

        # Stone331
        stone331 = Stone331Dataset(self.config)
        if len(stone331) > 0:
            n_train = max(5, len(stone331) // 3)
            train_st = [stone331[i] for i in range(n_train)]
            test_st = stone331
            self.run_experiment("Stone331", train_st, test_st, is_vessel=False)

        # CrackLS315 (visualization only)
        crackls315 = CrackLS315Dataset(self.config)
        if len(crackls315) > 0:
            print(f"\nCrackLS315: {len(crackls315)} images (visualization only)")
            self._generate_visualizations_only(crackls315, "CrackLS315")

        # CRKWH100 (visualization only)
        crkwh100 = CRKWH100Dataset(self.config)
        if len(crkwh100) > 0:
            print(f"\nCRKWH100: {len(crkwh100)} images (visualization only)")
            self._generate_visualizations_only(crkwh100, "CRKWH100")

        # SDNET Decks, Pavements, Walls
        for category in ['Decks', 'Pavements', 'Walls']:
            sdnet = SDNETDataset(self.config, category)
            if len(sdnet) > 0:
                self._generate_visualizations_only(sdnet, f"SDNET_{category}")

        # DRIVE
        train_drive = DRIVEDataset(self.config, 'training')
        test_drive = DRIVEDataset(self.config, 'test')
        if len(train_drive) > 0 and len(test_drive) > 0:
            self.run_experiment("DRIVE", train_drive, test_drive, is_vessel=True)

        # STARE (visualization only - no GT)
        stare = STAREDataset(self.config)
        if len(stare) > 0:
            self._generate_visualizations_only(stare, "STARE")

        # Save all results
        with open(os.path.join(self.config.output_dir, 'results', 'all_results.json'), 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 80)

        # Print summary table
        self._print_summary_table()

    def _generate_visualizations_only(self, dataset, name: str):
        """Generate visualizations for datasets without GT"""
        save_dir = os.path.join(self.config.output_dir, 'visualizations', name)
        os.makedirs(save_dir, exist_ok=True)

        # Train on pseudo-GT
        train_data = []
        for i in range(min(10, len(dataset))):
            img = dataset[i][0]
            features = FeatureExtractor(img, self.hed_model)
            hed_maps = [features.get(h) for h in HeuristicType.get_locked()]
            pseudo_gt = simple_average_fusion(hed_maps)
            train_data.append((img, pseudo_gt))

        evaluator = SolutionEvaluator(self.config, self.hed_model, self.config.use_learnable_fusion)
        ils = ILSHyperheuristic(self.config, evaluator)
        solution = ils.optimize(train_data, verbose=False)

        # Generate visualizations
        for i in range(min(20, len(dataset))):
            item = dataset[i]
            img = item[0]
            img_name = item[1] if len(item) > 1 else f"img_{i}"

            features = FeatureExtractor(img, self.hed_model)

            # Baseline
            hed_maps = [features.get(h) for h in HeuristicType.get_locked()]
            baseline_post = post_process(simple_average_fusion(hed_maps))

            # Enhanced
            all_maps = [features.get(h) for h in solution.all_heuristics]
            enhanced_post = post_process(simple_average_fusion(all_maps))

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            if img.shape[2] == 3:
                axes[0].imshow(img)
            else:
                axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(baseline_post, cmap='gray')
            axes[1].set_title('HED Baseline')
            axes[1].axis('off')

            axes[2].imshow(enhanced_post, cmap='gray')
            axes[2].set_title('ILS-HED (Ours)')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{img_name}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Saved {min(20, len(dataset))} visualizations to {save_dir}")

    def _print_summary_table(self):
        """Print summary table of all results"""
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Dataset':<15} {'Metric':<12} {'HED':<10} {'ILS-HED':<10} {'Improvement':<12}")
        print("-" * 60)

        for name, results in self.all_results.items():
            for metric, value in results['enhanced'].items():
                baseline = results['baseline'][metric]
                imp = value - baseline
                print(f"{name:<15} {metric:<12} {baseline:.4f}     {value:.4f}     {imp:+.4f}")
            print("-" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all()


if __name__ == "__main__":
    main()
