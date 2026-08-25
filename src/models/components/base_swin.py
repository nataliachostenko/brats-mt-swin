import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class BaseSwinUNETR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True
    ):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def load_pretrained(self, weights_path: str):
        print(f"Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = self.backbone.state_dict()

        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} of {len(model_dict)} layers.")
        if len(pretrained_dict) == 0:
            print("WARNING: no weights matched — check the checkpoint path.")

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """Returns the Swin encoder's hidden states before the decoder."""
        return self.backbone.swinViT(x, self.backbone.normalize)
