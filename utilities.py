"""
Utility functions for ILS-HED: image loading, ground-truth parsing, and metrics.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(img_path, as_gray=False):
    """
    Load an image from *img_path* using PIL, with an optional greyscale
    conversion.

    Args:
        img_path: Path to the image file.
        as_gray:  If True, return a 2-D greyscale array; otherwise return a
                  3-channel uint8 RGB array (RGBA images are converted to RGB).

    Returns:
        numpy.ndarray or None if the file cannot be opened.
    """
    try:
        img = Image.open(img_path)
        if as_gray:
            img = img.convert('L')
            return np.array(img)
        else:
            img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        print(f"Error loading image '{img_path}': {e}")
        return None


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_bsds500_gt(mat_path):
    """
    Load a BSDS500 ground-truth .mat file and return the union of all
    annotator boundary maps as a float32 array.

    Args:
        mat_path: Path to the .mat file.

    Returns:
        numpy.ndarray (H × W, float32) or None on failure.
    """
    try:
        mat = loadmat(mat_path)
        boundaries = []
        for i in range(mat['groundTruth'].shape[1]):
            item = mat['groundTruth'][0, i]
            if 'Boundaries' in item.dtype.names:
                boundaries.append(item['Boundaries'][0, 0].astype(np.float32))
        if boundaries:
            return np.maximum.reduce(boundaries)
    except Exception as e:
        print(f"Error loading BSDS500 ground truth '{mat_path}': {e}")
    return None


def load_binary_gt(gt_path):
    """
    Load a binary ground-truth image (BMP / PNG / GIF / …) and return a
    float32 mask where foreground pixels are 1.

    Args:
        gt_path: Path to the ground-truth image.

    Returns:
        numpy.ndarray (H × W, float32) or None on failure.
    """
    try:
        gt_img = Image.open(gt_path)
        gt = np.array(gt_img)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        return (gt > 0).astype(np.float32)
    except Exception as e:
        print(f"Error loading ground truth '{gt_path}': {e}")
        return None


def load_drive_gt(gt_path, mask_path=None):
    """
    Load a DRIVE ground-truth vessel map and optional FOV mask.

    Args:
        gt_path:   Path to the manual annotation file (.gif).
        mask_path: Optional path to the FOV mask file (.gif).

    Returns:
        Tuple (gt, mask) where each is a float32 / bool numpy array, or
        (None, None) on failure.
    """
    try:
        gt_img = Image.open(gt_path)
        gt = np.array(gt_img)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = (gt > 0).astype(np.float32)

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask > 0

        return gt, mask
    except Exception as e:
        print(f"Error loading DRIVE ground truth '{gt_path}': {e}")
        return None, None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_f1(pred, gt, threshold=0.5, epsilon=1e-7):
    """
    Compute the F1 score between a predicted probability map and a binary
    ground-truth mask.

    Args:
        pred:      Predicted edge probability map (H × W, float).
        gt:        Binary ground-truth map (H × W, float or bool).
        threshold: Binarisation threshold for *pred*.
        epsilon:   Small constant to avoid division by zero.

    Returns:
        float – F1 score in [0, 1].
    """
    binary_pred = (pred >= threshold).astype(np.float32)
    binary_gt = (gt > 0).astype(np.float32)

    tp = np.sum(binary_pred * binary_gt)
    fp = np.sum(binary_pred * (1 - binary_gt))
    fn = np.sum((1 - binary_pred) * binary_gt)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return 2 * precision * recall / (precision + recall + epsilon)


def compute_ods(preds, gts, thresholds=None, epsilon=1e-7):
    """
    Compute the Optimal Dataset Scale (ODS) F-measure by searching over a
    set of thresholds and picking the one that maximises the dataset-level F1.

    Args:
        preds:      List of predicted probability maps.
        gts:        List of binary ground-truth maps (same length as *preds*).
        thresholds: Iterable of candidate thresholds.  Defaults to 99 values
                    linearly spaced in (0, 1).
        epsilon:    Small constant for numerical stability.

    Returns:
        Tuple (best_threshold, best_f1).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_f1, best_thresh = 0.0, 0.5
    for thresh in thresholds:
        f1_scores = [compute_f1(p, g, threshold=thresh, epsilon=epsilon)
                     for p, g in zip(preds, gts)]
        mean_f1 = float(np.mean(f1_scores))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thresh = thresh

    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    """Create *path* (and any missing parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def normalise(array, epsilon=1e-8):
    """
    Min-max normalise *array* to [0, 1].

    Args:
        array:   Input numpy array.
        epsilon: Guard against division by zero when the array is constant.

    Returns:
        Normalised float32 array.
    """
    arr = array.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val < epsilon:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)
