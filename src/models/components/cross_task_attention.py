import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTaskAttention3D(nn.Module):
    def __init__(self, seg_channels: int, cls_channels: int, reduction_ratio: int = 4):
        """
        Uses the classifier's feature vector to drive channel attention over the
        segmentation decoder's activation maps.
        """
        super().__init__()

        self.cls_projection = nn.Sequential(
            nn.Linear(cls_channels, seg_channels // reduction_ratio, bias=False),
            nn.LayerNorm(seg_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(seg_channels // reduction_ratio, seg_channels, bias=False),
            nn.Sigmoid()
        )

        self.seg_conv = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1, groups=seg_channels),
            nn.InstanceNorm3d(seg_channels),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=1),
            nn.InstanceNorm3d(seg_channels)
        )

    def forward(self, x_seg: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        identity = x_seg
        x_seg = self.seg_conv(x_seg)

        attention_weights = self.cls_projection(x_cls)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # classifier gates out channels unrelated to the detected pathology
        attended_features = x_seg * attention_weights

        out = self.fusion(attended_features) + identity
        return F.gelu(out)
