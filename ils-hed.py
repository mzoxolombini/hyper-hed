#!/usr/bin/env python3
"""
ILS Hyperheuristic for HED Edge Detection Enhancement
Complete implementation for all datasets:
- BSDS500 (edge detection)
- DeepCrack (crack detection)
- Stone331 (crack detection with masks)
- CrackLS315 (crack detection)
- CRKWH100 (crack detection)
- SDNET (crack detection - Decks, Pavements, Walls)
- DRIVE (retinal vessel segmentation)
- STARE (retinal vessel segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, sobel
from skimage import io, filters, morphology
from skimage import feature as skfeature
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
import os
import random
from enum import Enum
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from tqdm import tqdm
import json
from glob import glob
from PIL import Image

warnings.filterwarnings('ignore')


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
# HED NETWORK (Locked Core)
# ============================================================================

class HEDNetwork(nn.Module):
    """Official HED architecture with 5 side outputs (S1-S5) - LOCKED"""

    def __init__(self, pretrained_path: Optional[str] = None):
        super(HEDNetwork, self).__init__()

        # VGG-16 stages
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Side outputs
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)

        self.fuse_weight = nn.Parameter(torch.ones(5) / 5)
        self._initialize_weights()

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in [self.dsn1, self.dsn2, self.dsn3, self.dsn4, self.dsn5]:
                    nn.init.normal_(m.weight, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded HED weights: {len(pretrained_dict)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"Warning: Could not load HED weights: {e}")

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        d1 = F.interpolate(self.dsn1(conv1), size=(h, w), mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dsn2(conv2), size=(h, w), mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.dsn3(conv3), size=(h, w), mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.dsn4(conv4), size=(h, w), mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.dsn5(conv5), size=(h, w), mode='bilinear', align_corners=False)

        d1, d2, d3, d4, d5 = map(torch.sigmoid, [d1, d2, d3, d4, d5])

        fuse_weights = F.softmax(self.fuse_weight, dim=0)
        fuse = d1 * fuse_weights[0] + d2 * fuse_weights[1] + d3 * fuse_weights[2] + \
               d4 * fuse_weights[3] + d5 * fuse_weights[4]
        fuse = torch.sigmoid(fuse)

        return [d1, d2, d3, d4, d5, fuse]

    @torch.no_grad()
    def extract_side_outputs(self, image_np: np.ndarray) -> List[np.ndarray]:
        """Extract the 5 side outputs (S1-S5) - LOCKED features"""
        self.eval()
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        if image_np.max() > 1:
            image_np = image_np / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std

        image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(next(self.parameters()).device)

        outputs = self.forward(image_tensor)
        return [o.squeeze().cpu().numpy() for o in outputs[:5]]


# ============================================================================
# TRADITIONAL DETECTORS (Variable Heuristics)
# ============================================================================

class TraditionalDetectors:
    """Classical edge detectors - VARIABLE heuristics"""

    @staticmethod
    def canny(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        if len(image.shape) > 2:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image
        smoothed = gaussian_filter(gray, sigma=sigma)
        edges = skfeature.canny(smoothed, sigma=sigma, low_threshold=0.05, high_threshold=0.15)
        return edges.astype(np.float32)

    @staticmethod
    def sobel(image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return (magnitude / magnitude.max()).astype(np.float32) if magnitude.max() > 0 else magnitude.astype(np.float32)

    @staticmethod
    def laplacian(image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image
        lap = filters.laplace(gray)
        lap = np.abs(lap)
        return (lap / lap.max()).astype(np.float32) if lap.max() > 0 else lap.astype(np.float32)

    @staticmethod
    def gabor(image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = image

        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        responses = []

        for theta in orientations:
            gabor_real = filters.gabor(gray, frequency=0.1, theta=theta, sigma_x=2.0, sigma_y=2.0)[0]
            responses.append(np.abs(gabor_real))

        combined = np.max(responses, axis=0)
        return (combined / combined.max()).astype(np.float32) if combined.max() > 0 else combined.astype(np.float32)


# ============================================================================
# HEURISTIC TYPES
# ============================================================================

class HeuristicType(Enum):
    # LOCKED (always present)
    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    S4 = "s4"
    S5 = "s5"

    # VARIABLE (optional)
    CANNY = "canny"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    GABOR = "gabor"

    @classmethod
    def get_locked(cls) -> Set:
        return {cls.S1, cls.S2, cls.S3, cls.S4, cls.S5}

    @classmethod
    def get_variable(cls) -> Set:
        return {cls.CANNY, cls.SOBEL, cls.LAPLACIAN, cls.GABOR}

    def get_extractor(self):
        mapping = {
            HeuristicType.CANNY: TraditionalDetectors.canny,
            HeuristicType.SOBEL: TraditionalDetectors.sobel,
            HeuristicType.LAPLACIAN: TraditionalDetectors.laplacian,
            HeuristicType.GABOR: TraditionalDetectors.gabor,
        }
        return mapping.get(self)


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """Extracts and caches features from both locked and variable detectors"""

    def __init__(self, image: np.ndarray, hed_model: HEDNetwork):
        self.image = image
        self.hed_model = hed_model
        self._cache = {}
        self._extract_all()

    def _extract_all(self):
        # Extract locked HED features (S1-S5)
        hed_outputs = self.hed_model.extract_side_outputs(self.image)
        for h, output in zip(HeuristicType.get_locked(), hed_outputs):
            self._cache[h] = output

        # Prepare grayscale
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor((self.image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        else:
            gray = self.image
        self.gray_image = gray

        # Extract variable traditional features
        for h in HeuristicType.get_variable():
            extractor = h.get_extractor()
            if extractor:
                self._cache[h] = extractor(self.gray_image)

    def get(self, heuristic: HeuristicType) -> np.ndarray:
        return self._cache.get(heuristic)

    def get_multiple(self, heuristics: Set[HeuristicType]) -> List[np.ndarray]:
        return [self.get(h) for h in heuristics]


# ============================================================================
# SOLUTION CLASS
# ============================================================================

@dataclass
class Solution:
    """Solution: locked S1-S5 + variable traditional detectors"""
    variable_heuristics: Set[HeuristicType] = field(default_factory=set)
    metric: float = 0.0
    fusion_weights: Optional[np.ndarray] = None

    @property
    def locked_heuristics(self) -> Set[HeuristicType]:
        return HeuristicType.get_locked()

    @property
    def all_heuristics(self) -> Set[HeuristicType]:
        return self.locked_heuristics | self.variable_heuristics

    def to_dict(self) -> Dict:
        return {
            'variable': [h.value for h in self.variable_heuristics],
            'metric': self.metric,
            'weights': self.fusion_weights.tolist() if self.fusion_weights is not None else None
        }

    def copy(self):
        return Solution(
            variable_heuristics=self.variable_heuristics.copy(),
            metric=self.metric,
            fusion_weights=self.fusion_weights.copy() if self.fusion_weights is not None else None
        )

    def __str__(self) -> str:
        var_str = ', '.join([h.value for h in self.variable_heuristics])
        return f"Solution(variable=[{var_str}], metric={self.metric:.4f})"


# ============================================================================
# FUSION MODULE
# ============================================================================

class FusionModule(nn.Module):
    """Learnable fusion module"""

    def __init__(self, n_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs) / n_inputs)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, edge_maps: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(edge_maps, dim=0)
        weights = F.softmax(self.weights / self.temperature, dim=0)
        fused = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        return torch.sigmoid(fused)

    def get_weights(self) -> np.ndarray:
        with torch.no_grad():
            return F.softmax(self.weights / self.temperature, dim=0).cpu().numpy()


def simple_average_fusion(edge_maps: List[np.ndarray]) -> np.ndarray:
    return np.mean(edge_maps, axis=0)


def post_process(edge_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid -> Threshold -> Skeletonize"""
    activated = 1.0 / (1.0 + np.exp(-np.clip(2 * edge_map - 1, -50, 50)))
    binary = (activated > threshold).astype(np.uint8)
    from skimage.morphology import skeletonize
    thin = skeletonize(binary).astype(np.float32)
    return thin


# ============================================================================
# SOLUTION EVALUATOR
# ============================================================================

class SolutionEvaluator:
    """Evaluates solutions using HED weighted binary cross-entropy (higher is better)"""

    def __init__(self, config: Config, hed_model: HEDNetwork, use_learnable_fusion: bool = True):
        self.config = config
        self.hed_model = hed_model
        self.use_learnable_fusion = use_learnable_fusion
        self.fusion_module = None

    def evaluate(self, solution: Solution, train_data: List[Tuple[np.ndarray, np.ndarray]],
                 num_epochs: int = 5) -> float:
        if self.use_learnable_fusion:
            return self._evaluate_with_learnable_fusion(solution, train_data, num_epochs)
        else:
            return self._evaluate_with_average_fusion(solution, train_data, num_epochs)

    def _evaluate_with_average_fusion(self, solution: Solution,
                                      train_data: List[Tuple[np.ndarray, np.ndarray]],
                                      num_epochs: int) -> float:
        all_heuristics = list(solution.all_heuristics)
        total_metric = 0

        for epoch in range(num_epochs):
            epoch_metric = 0
            for img, gt in train_data:
                features = FeatureExtractor(img, self.hed_model)
                edge_maps = features.get_multiple(all_heuristics)
                fused = simple_average_fusion(edge_maps)
                metric = self._compute_metric(fused, gt)
                epoch_metric += metric

            total_metric += epoch_metric / len(train_data)

        solution.metric = total_metric / num_epochs
        return solution.metric

    def _evaluate_with_learnable_fusion(self, solution: Solution,
                                        train_data: List[Tuple[np.ndarray, np.ndarray]],
                                        num_epochs: int) -> float:
        all_heuristics = list(solution.all_heuristics)

        self.fusion_module = FusionModule(len(all_heuristics)).to(self.config.device)
        optimizer = torch.optim.Adam(self.fusion_module.parameters(), lr=0.01)

        total_metric = 0

        for epoch in range(num_epochs):
            epoch_metric = 0
            for img, gt in train_data:
                features = FeatureExtractor(img, self.hed_model)
                edge_maps = [torch.from_numpy(features.get(h)).float().to(self.config.device)
                             for h in all_heuristics]
                gt_tensor = torch.from_numpy(gt).float().to(self.config.device)

                fused = self.fusion_module(edge_maps)
                metric = self._compute_metric_tensor(fused, gt_tensor)

                loss = -metric
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_metric += metric.item()

            total_metric += epoch_metric / len(train_data)

        solution.fusion_weights = self.fusion_module.get_weights()
        solution.metric = total_metric / num_epochs
        return solution.metric

    def _compute_metric(self, pred: np.ndarray, gt: np.ndarray, epsilon: float = 1e-7) -> float:
        pred = np.clip(pred, epsilon, 1 - epsilon)
        beta = np.mean(gt)
        w_pos = 1.0 / (beta + epsilon)
        w_neg = 1.0 / (1 - beta + epsilon)
        loss = -np.mean(w_neg * (1 - gt) * np.log(1 - pred) + w_pos * gt * np.log(pred))
        return -loss

    def _compute_metric_tensor(self, pred: torch.Tensor, gt: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        beta = torch.mean(gt)
        w_pos = 1.0 / (beta + epsilon)
        w_neg = 1.0 / (1 - beta + epsilon)
        loss = -torch.mean(w_neg * (1 - gt) * torch.log(1 - pred) + w_pos * gt * torch.log(pred))
        return -loss


# ============================================================================
# ILS HYPERHEURISTIC
# ============================================================================

class ILSHyperheuristic:
    """Iterative Local Search - only perturbs variable heuristics"""

    def __init__(self, config: Config, evaluator: SolutionEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.all_variable = list(HeuristicType.get_variable())

    def generate_initial_solution(self) -> Solution:
        n = random.randint(self.config.min_traditional, len(self.all_variable))
        variable_set = set(random.sample(self.all_variable, n))
        return Solution(variable_heuristics=variable_set, metric=0.0)

    def perturb(self, solution: Solution) -> Solution:
        variable_list = list(solution.variable_heuristics)
        unused = [h for h in self.all_variable if h not in variable_list]

        operators = []
        if unused:
            operators.append('add')
        if len(variable_list) > self.config.min_traditional:
            operators.append('remove')
        if unused and len(variable_list) > 0:
            operators.append('swap')

        if not operators:
            return solution.copy()

        operator = random.choice(operators)
        new_variable = set(variable_list)

        if operator == 'add':
            new_variable.add(random.choice(unused))
        elif operator == 'remove':
            new_variable.remove(random.choice(variable_list))
        elif operator == 'swap':
            new_variable.remove(random.choice(variable_list))
            new_variable.add(random.choice(unused))

        return Solution(variable_heuristics=new_variable, metric=0.0)

    def optimize(self, train_data: List[Tuple[np.ndarray, np.ndarray]],
                 verbose: bool = True) -> Solution:
        best_overall = None

        for restart in range(self.config.num_restarts):
            if verbose:
                print(f"  Restart {restart + 1}/{self.config.num_restarts}")

            current = self.generate_initial_solution()
            current_metric = self.evaluator.evaluate(current, train_data, self.config.num_epochs)
            current.metric = current_metric

            if best_overall is None or current_metric > best_overall.metric:
                best_overall = current.copy()
                if verbose:
                    print(f"    New best: {best_overall}")

            no_improve = 0

            for iteration in range(self.config.max_iter):
                candidate = self.perturb(current)

                if candidate.variable_heuristics == current.variable_heuristics:
                    continue

                candidate_metric = self.evaluator.evaluate(candidate, train_data, self.config.num_epochs)
                candidate.metric = candidate_metric

                if candidate_metric > current.metric:
                    current = candidate
                    no_improve = 0

                    if candidate_metric > best_overall.metric:
                        best_overall = candidate.copy()
                        if verbose:
                            print(f"    New best (iter {iteration}): {best_overall}")
                else:
                    no_improve += 1

                if no_improve >= self.config.patience:
                    break

        return best_overall


# ============================================================================
# DATASET LOADERS
# ============================================================================

class BSDS500Dataset:
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.split = split
        self.samples = []

        img_dir = os.path.join(config.bsds500_root, 'images', split)
        gt_dir = os.path.join(config.bsds500_root, 'ground_truth', split)

        if not os.path.exists(img_dir):
            print(f"BSDS500 {split} not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, base + '.mat')
            self.samples.append({'image': img_path, 'gt': gt_path if os.path.exists(gt_path) else None, 'name': base})

        print(f"BSDS500 {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            try:
                mat = loadmat(sample['gt'])
                if 'groundTruth' in mat:
                    boundaries = []
                    for i in range(mat['groundTruth'].shape[1]):
                        item = mat['groundTruth'][0, i]
                        if 'Boundaries' in item.dtype.names:
                            boundaries.append(item['Boundaries'][0, 0].astype(np.float32))
                    if boundaries:
                        gt = np.maximum.reduce(boundaries)
                        gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)
            except:
                pass

        return img, gt, sample['name'], orig


class DeepCrackDataset:
    def __init__(self, config: Config):
        self.config = config
        self.samples = []

        img_dir = os.path.join(config.deepcrack_root, 'image')
        gt_dir = os.path.join(config.deepcrack_root, 'ground_truth')

        if not os.path.exists(img_dir):
            print(f"DeepCrack not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.JPG'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, base + '.bmp')
            self.samples.append({'image': img_path, 'gt': gt_path if os.path.exists(gt_path) else None, 'name': base})

        print(f"DeepCrack: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            gt_img = io.imread(sample['gt'], as_gray=True)
            gt = (gt_img > 0).astype(np.float32)
            gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)

        return img, gt, sample['name'], orig


class Stone331Dataset:
    def __init__(self, config: Config):
        self.config = config
        self.samples = []

        masks = {}
        if os.path.exists(config.stone331_mask_root):
            for mask_file in glob(os.path.join(config.stone331_mask_root, '*.bmp')):
                base = os.path.splitext(os.path.basename(mask_file))[0]
                masks[base] = mask_file

        for img_path in sorted(glob(os.path.join(config.stone331_root, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'gt': masks.get(base), 'name': base})

        print(f"Stone331: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)

        gt = None
        if sample['gt']:
            gt_img = io.imread(sample['gt'], as_gray=True)
            gt = (gt_img > 0).astype(np.float32)
            gt = cv2.resize(gt, self.config.image_size, interpolation=cv2.INTER_NEAREST)

        return img, gt, sample['name'], orig


class CrackLS315Dataset:
    def __init__(self, config: Config):
        self.config = config
        self.samples = []

        for img_path in sorted(glob(os.path.join(config.crackls315_root, '*.jpg'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"CrackLS315: {len(self.samples)} images (no GT)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, None, sample['name'], orig


class CRKWH100Dataset:
    def __init__(self, config: Config):
        self.config = config
        self.samples = []

        for img_path in sorted(glob(os.path.join(config.crkwh100_root, '*.png'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"CRKWH100: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, None, sample['name'], orig


class SDNETDataset:
    def __init__(self, config: Config, category: str = 'Decks'):
        self.config = config
        self.samples = []

        cracked_dir = os.path.join(config.sdnet_root, category, 'Cracked')

        if os.path.exists(cracked_dir):
            for img_file in os.listdir(cracked_dir):
                if img_file.lower().endswith(('.jpg', '.png')):
                    self.samples.append({
                        'path': os.path.join(cracked_dir, img_file),
                        'name': img_file,
                        'category': category
                    })

        print(f"SDNET {category}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['path'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        return img, sample['name'], orig, sample['category']


class DRIVEDataset:
    def __init__(self, config: Config, split: str = 'test'):
        self.config = config
        self.split = split
        self.samples = []

        if split == 'test':
            img_dir = os.path.join(config.drive_root, 'test', 'images')
            mask_dir = os.path.join(config.drive_root, 'test', 'mask')
            gt_dir = None
        else:
            img_dir = os.path.join(config.drive_root, 'training', 'images')
            mask_dir = os.path.join(config.drive_root, 'training', 'mask')
            gt_dir = os.path.join(config.drive_root, 'training', '1st_manual')

        if not os.path.exists(img_dir):
            print(f"DRIVE {split} not found")
            return

        for img_path in sorted(glob(os.path.join(img_dir, '*.tif')) + glob(os.path.join(img_dir, '*.tiff'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            base_clean = base.replace('_training', '').replace('_test', '')

            mask_path = os.path.join(mask_dir, f"{base_clean}_mask.gif") if mask_dir else None
            if not os.path.exists(mask_path):
                mask_path = None

            gt_path = os.path.join(gt_dir, f"{base_clean}_manual1.gif") if gt_dir else None
            if gt_path and not os.path.exists(gt_path):
                gt_path = None

            self.samples.append({'image': img_path, 'mask': mask_path, 'gt': gt_path, 'name': base_clean})

        print(f"DRIVE {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image with PIL for LZW compression
        img_pil = Image.open(sample['image'])
        img = np.array(img_pil)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        if img.max() > 1:
            img = img / 255.0

        # Load FOV mask
        fov_mask = np.ones(img.shape[:2], dtype=bool)
        if sample['mask']:
            mask = io.imread(sample['mask'])
            fov_mask = (mask > 0)

        # Load GT
        gt = np.zeros(img.shape[:2], dtype=np.float32)
        if sample['gt']:
            gt_img = io.imread(sample['gt'])
            gt = (gt_img > 0).astype(np.float32)

        # Resize
        img = cv2.resize(img, self.config.drive_size, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, self.config.drive_size, interpolation=cv2.INTER_NEAREST)
        fov_mask = cv2.resize(fov_mask.astype(np.uint8), self.config.drive_size,
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        return img, gt, fov_mask, sample['name']


class STAREDataset:
    def __init__(self, config: Config):
        self.config = config
        self.samples = []

        # STARE has images in root directory
        for img_path in sorted(glob(os.path.join(config.stare_root, '*.ppm'))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            self.samples.append({'image': img_path, 'name': base})

        print(f"STARE: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = io.imread(sample['image'])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() > 1:
            img = img / 255.0
        orig = img.copy()
        img = cv2.resize(img, self.config.stare_size, interpolation=cv2.INTER_LINEAR)
        return img, sample['name'], orig


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class Metrics:
    @staticmethod
    def compute_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Tuple[float, float, float]:
        binary = (pred >= threshold).astype(np.float32)
        tp = np.sum(binary * gt)
        fp = np.sum(binary * (1 - gt))
        fn = np.sum((1 - binary) * gt)
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * prec * rec / (prec + rec + 1e-7)
        return f1, prec, rec

    @staticmethod
    def compute_iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
        binary = (pred >= threshold).astype(np.float32)
        intersection = np.sum(binary * gt)
        union = np.sum(np.maximum(binary, gt))
        return intersection / (union + 1e-7)

    @staticmethod
    def compute_sensitivity_specificity(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None,
                                        threshold: float = 0.5) -> Tuple[float, float]:
        binary = (pred >= threshold).astype(np.float32)
        if mask is not None:
            binary = binary * mask
            gt = gt * mask
        tp = np.sum(binary * gt)
        tn = np.sum((1 - binary) * (1 - gt))
        fp = np.sum(binary * (1 - gt))
        fn = np.sum((1 - binary) * gt)
        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)
        return sens, spec


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
            print(f"No GT available, using pseudo-GT from HED")
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
            # Split into train/test
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