import torch
import torch.nn as nn
from src.models.components.base_swin import BaseSwinUNETR
from src.models.components.cross_task_attention import CrossTaskAttention3D


class MultiTaskSwinUNETR(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, feature_size=48):
        super().__init__()

        self.backbone = BaseSwinUNETR(
            in_channels=in_channels,
            out_channels=32,
            feature_size=feature_size,
            use_checkpoint=True
        )

        # bottleneck has feature_size * 16 channels (48 * 16 = 768)
        bottleneck_dim = feature_size * 16
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.cls_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.cls_out = nn.Linear(128, 3)  # NCR, ED, ET presence logits

        self.cross_attention = CrossTaskAttention3D(
            seg_channels=32,
            cls_channels=128,
            reduction_ratio=4
        )

        self.final_seg = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        hidden_states = self.backbone.extract_features(x)
        bottleneck = hidden_states[-1]  # (B, 768, D/32, H/32, W/32)

        pooled = self.global_pool(bottleneck).flatten(1)
        cls_features = self.cls_head(pooled)
        cls_logits = self.cls_out(cls_features)

        seg_features = self.backbone(x)
        attended_features = self.cross_attention(seg_features, cls_features)
        seg_logits = self.final_seg(attended_features)

        return seg_logits, cls_logits
