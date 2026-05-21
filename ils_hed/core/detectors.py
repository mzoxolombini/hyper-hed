"""
Classical edge detectors (variable heuristics) for ILS-HED.
Contains: DetectorConfig dataclass, TraditionalDetectors, HeuristicType enum.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
from skimage import filters
from skimage import feature as skfeature
from dataclasses import dataclass
from typing import Set
from enum import Enum


@dataclass
class DetectorConfig:
    """Configuration for a single detector instance."""
    name: str
    params: dict


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
