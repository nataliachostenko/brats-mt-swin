import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete


class BraTSLightningModule(pl.LightningModule):
    """Lightning wrapper: network, loss, optimizer and metrics for the BraTS task."""

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        roi_size: tuple = (96, 96, 96)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = net

        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
        self.val_dice = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["seg"]

        out = self(images)
        if isinstance(out, tuple):
            seg_logits, cls_logits = out
            is_multitask = True
        else:
            seg_logits = out
            is_multitask = False

        loss_seg = self.loss_fn(seg_logits, labels)

        if is_multitask:
            has_ncr = (labels == 1).sum(dim=[1, 2, 3, 4]) > 0
            has_ed = (labels == 2).sum(dim=[1, 2, 3, 4]) > 0
            has_et = (labels == 3).sum(dim=[1, 2, 3, 4]) > 0
            cls_targets = torch.stack([has_ncr, has_ed, has_et], dim=1).float()

            # multi-label (components can co-occur), not multi-class
            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
            total_loss = loss_seg + loss_cls

            self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss_seg", loss_seg, on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=False, sync_dist=True)
        else:
            total_loss = loss_seg
            self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss_seg", loss_seg, on_step=True, on_epoch=False, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["seg"]

        def predictor_wrapper(x):
            out = self.net(x)
            return out[0] if isinstance(out, tuple) else out

        logits = sliding_window_inference(
            inputs=images,
            roi_size=self.hparams.roi_size,
            sw_batch_size=4,
            predictor=predictor_wrapper,
            overlap=0.5
        )

        loss = self.loss_fn(logits, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        from monai.data import decollate_batch
        val_outputs = [self.post_pred(i) for i in decollate_batch(logits)]
        val_labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.val_dice(y_pred=val_outputs, y=val_labels)

    def on_validation_epoch_end(self):
        class_dice = self.val_dice.aggregate()
        dices = {
            "val/dice_NCR": class_dice[0].item(),
            "val/dice_ED": class_dice[1].item(),
            "val/dice_ET": class_dice[2].item(),
            "val/dice_mean": class_dice.mean().item()
        }
        self.val_dice.reset()
        self.log_dict(dices, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
