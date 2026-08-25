import torch
import torch.nn as nn
from src.models.components.base_swin import BaseSwinUNETR
from src.models.components.cross_task_attention import CrossTaskAttention3D


class MultiTaskSwinUNETRDetach(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, feature_size=48):
        super().__init__()

        self.backbone = BaseSwinUNETR(
            in_channels=in_channels,
            out_channels=32,
            feature_size=feature_size,
            use_checkpoint=True
        )

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
        bottleneck = hidden_states[-1]

        # detach: classification loss must not shape the shared encoder
        pooled = self.global_pool(bottleneck).flatten(1)
        pooled_detached = pooled.detach()
        cls_features = self.cls_head(pooled_detached)
        cls_logits = self.cls_out(cls_features)

        seg_features = self.backbone(x)

        # detach: segmentation loss must not shape the classification head
        cls_features_detached = cls_features.detach()
        attended_features = self.cross_attention(seg_features, cls_features_detached)
        seg_logits = self.final_seg(attended_features)

        return seg_logits, cls_logits
