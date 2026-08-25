import torch
from src.models.components.cross_task_attention import CrossTaskAttention3D


def test_cta_output_shape():
    cta = CrossTaskAttention3D(seg_channels=16, cls_channels=32, reduction_ratio=4)
    x_seg = torch.randn(2, 16, 8, 8, 8)
    x_cls = torch.randn(2, 32)

    out = cta(x_seg, x_cls)

    assert out.shape == x_seg.shape


def test_cta_is_not_identity():
    cta = CrossTaskAttention3D(seg_channels=16, cls_channels=32, reduction_ratio=4)
    x_seg = torch.randn(2, 16, 8, 8, 8)
    x_cls = torch.randn(2, 32)

    out = cta(x_seg, x_cls)

    assert not torch.allclose(out, x_seg)
