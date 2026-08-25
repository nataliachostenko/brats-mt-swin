import torch
from monai.transforms import MapLabelValued


def test_brats_label_remap():
    # BraTS masks use {0,1,2,4}; the pipeline remaps them to contiguous {0,1,2,3}
    remap = MapLabelValued(keys=["seg"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3])
    seg = torch.tensor([[0, 1], [2, 4]])

    out = remap({"seg": seg})["seg"]

    assert torch.equal(out, torch.tensor([[0, 1], [2, 3]]))
    assert out.max() == 3
