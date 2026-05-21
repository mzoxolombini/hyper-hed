"""
Evaluation metrics for ILS-HED.
Contains: Metrics class (F1, IoU, sensitivity/specificity).
"""

import numpy as np
from typing import Optional, Tuple


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
    def compute_sensitivity_specificity(pred: np.ndarray, gt: np.ndarray,
                                        mask: Optional[np.ndarray] = None,
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
