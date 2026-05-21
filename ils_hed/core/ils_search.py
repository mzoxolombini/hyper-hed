"""
Iterative Local Search hyper-heuristic for ILS-HED.
Contains: Solution dataclass, FeatureExtractor, SolutionEvaluator, ILSHyperheuristic.
"""

import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import cv2

from .detectors import HeuristicType, TraditionalDetectors
from .hed_model import HEDNetwork
from .fusion import FusionModule, simple_average_fusion


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

    def get_multiple(self, heuristics) -> List[np.ndarray]:
        return [self.get(h) for h in heuristics]


# ============================================================================
# SOLUTION EVALUATOR
# ============================================================================

class SolutionEvaluator:
    """Evaluates solutions using HED weighted binary cross-entropy (higher is better)"""

    def __init__(self, config, hed_model: HEDNetwork, use_learnable_fusion: bool = True):
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

    def __init__(self, config, evaluator: SolutionEvaluator):
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
