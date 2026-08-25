import numpy as np
from src.utils.postprocess import remove_small_islands


def test_removes_small_island():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0, 0, 0] = True          # isolated single-voxel island
    mask[5:8, 5:8, 5:8] = True    # 27-voxel island

    filtered = remove_small_islands(mask, min_size=10)

    assert not filtered[0, 0, 0]
    assert filtered[5:8, 5:8, 5:8].all()


def test_min_size_zero_is_noop():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = True

    filtered = remove_small_islands(mask, min_size=0)

    assert filtered[0, 0, 0]


def test_empty_mask_returns_empty():
    mask = np.zeros((5, 5, 5), dtype=bool)

    filtered = remove_small_islands(mask, min_size=1)

    assert not filtered.any()
