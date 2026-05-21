"""
Fusion module for ILS-HED.
Contains: FusionModule (learnable weighted fusion), helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from skimage.morphology import skeletonize


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
    thin = skeletonize(binary).astype(np.float32)
    return thin
