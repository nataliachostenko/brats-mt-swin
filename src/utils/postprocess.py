import numpy as np
from scipy.ndimage import label


def remove_small_islands(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than min_size voxels from a binary mask."""
    if min_size <= 0:
        return mask.astype(bool)

    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled_mask.ravel())
    keep_components = component_sizes >= min_size
    keep_components[0] = False  # background is never a component to keep

    return keep_components[labeled_mask]
