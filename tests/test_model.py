import torch
from src.models.components.multi_task_swin import MultiTaskSwinUNETR


def test_multitask_forward_shapes():
    model = MultiTaskSwinUNETR(in_channels=4, num_classes=4, feature_size=12)
    x = torch.randn(2, 4, 64, 64, 64)

    seg_logits, cls_logits = model(x)

    assert seg_logits.shape == (2, 4, 64, 64, 64)
    assert cls_logits.shape == (2, 3)
