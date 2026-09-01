import torch
import sys
import os
from pathlib import Path
import typing

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.models.components.multi_task_swin import MultiTaskSwinUNETR
from src.models.brats_module import BraTSLightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from tqdm import tqdm
import omegaconf
import omegaconf.base

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any
])

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def convert_to_regions(y):
    if y.shape[0] > 1:
        y = torch.argmax(y, dim=0, keepdim=True)

    wt = (y == 1) | (y == 2) | (y == 3)
    tc = (y == 1) | (y == 3)
    et = (y == 3)

    return torch.cat([wt, tc, et], dim=0).float()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    CHECKPOINT_PATH = os.environ.get(
        "CHECKPOINT_PATH",
        "/work/ab0995/a270263/mgr/logs/brats-mgr-project/unwx02xp/checkpoints/resume.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating region-wise on: {device}")

    print("Initializing dataloader")
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    print("Loading model")
    net = MultiTaskSwinUNETR(in_channels=4, num_classes=4, feature_size=48)
    model = BraTSLightningModule.load_from_checkpoint(CHECKPOINT_PATH, net=net, map_location=device)
    model.to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    precision_lists = {"WT": [], "TC": [], "ET": []}
    recall_lists = {"WT": [], "TC": [], "ET": []}

    print("Processing patients")

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch["image"].to(device)
            labels = batch["seg"].to(device)

            def predictor_wrapper(x):
                out = model.net(x)
                return out[0] if isinstance(out, tuple) else out

            logits = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=predictor_wrapper,
                overlap=0.5
            )

            preds = [post_trans(i) for i in decollate_batch(logits)]

            region_preds = [convert_to_regions(p) for p in preds]
            region_labels = [convert_to_regions(l) for l in decollate_batch(labels)]

            dice_metric(y_pred=region_preds, y=region_labels)

            for p_pred, p_true in zip(region_preds, region_labels):
                for idx, region_name in enumerate(["WT", "TC", "ET"]):
                    pred_mask = p_pred[idx] > 0.5
                    true_mask = p_true[idx] > 0.5

                    tp = torch.sum(pred_mask & true_mask).item()
                    fp = torch.sum(pred_mask & ~true_mask).item()
                    fn = torch.sum(~pred_mask & true_mask).item()

                    if (tp + fn) > 0:
                        rec = tp / (tp + fn)
                    else:
                        rec = float('nan')

                    if (tp + fp) > 0:
                        prec = tp / (tp + fp)
                    else:
                        prec = float('nan')

                    precision_lists[region_name].append(prec)
                    recall_lists[region_name].append(rec)

    raw_scores = dice_metric.aggregate()

    if isinstance(raw_scores, (tuple, list)):
        raw_scores = raw_scores[0]

    raw_scores = raw_scores.view(-1, 3)

    wt_scores = raw_scores[:, 0]
    tc_scores = raw_scores[:, 1]
    et_scores = raw_scores[:, 2]

    wt_mean = wt_scores[~torch.isnan(wt_scores)].mean().item()
    tc_mean = tc_scores[~torch.isnan(tc_scores)].mean().item()
    et_mean = et_scores[~torch.isnan(et_scores)].mean().item()

    print("\n" + "=" * 40)
    print("FINAL RESULTS (REGION-WISE):")
    print("=" * 40)

    for idx, region_name in enumerate(["WT", "TC", "ET"]):
        r_scores = raw_scores[:, idx]
        dice_m = r_scores[~torch.isnan(r_scores)].mean().item()

        p_list = precision_lists[region_name]
        p_tensor = torch.tensor(p_list)
        prec_m = p_tensor[~torch.isnan(p_tensor)].mean().item()

        r_list = recall_lists[region_name]
        r_tensor = torch.tensor(r_list)
        rec_m = r_tensor[~torch.isnan(r_tensor)].mean().item()

        print(f"Region: {region_name}")
        print(f"  Dice:      {dice_m:.4f}")
        print(f"  Precision: {prec_m:.4f} (n={len(p_list) - torch.isnan(p_tensor).sum().item()} patients)")
        print(f"  Recall:    {rec_m:.4f} (n={len(r_list) - torch.isnan(r_tensor).sum().item()} patients)")
        print("-" * 40)

    print(f"MEAN BraTS Dice: {(wt_mean + tc_mean + et_mean) / 3:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
