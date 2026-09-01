import torch
import sys
import os
from pathlib import Path
import typing
import numpy as np
from sklearn.metrics import average_precision_score

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.models.components.multi_task_swin import MultiTaskSwinUNETR
from src.models.brats_module import BraTSLightningModule
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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


def convert_to_region_probs(prob):
    # prob has shape (4, H, W, D) after softmax
    wt_prob = prob[1] + prob[2] + prob[3]
    tc_prob = prob[1] + prob[3]
    et_prob = prob[3]
    return torch.stack([wt_prob, tc_prob, et_prob], dim=0)


def convert_to_region_binary(prob):
    class_map = torch.argmax(prob, dim=0)
    wt_bin = (class_map == 1) | (class_map == 2) | (class_map == 3)
    tc_bin = (class_map == 1) | (class_map == 3)
    et_bin = (class_map == 3)
    return torch.stack([wt_bin, tc_bin, et_bin], dim=0).float()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    CHECKPOINT_PATH = os.environ.get(
        "CHECKPOINT_PATH",
        "/work/ab0995/a270263/mgr/logs/brats-mgr-project/unwx02xp/checkpoints/resume.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating (Precision, Recall, PR AUC) on: {device}")

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
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="none", get_not_nans=False)

    precision_lists = {"WT": [], "TC": [], "ET": []}
    recall_lists = {"WT": [], "TC": [], "ET": []}
    pr_auc_lists = {"WT": [], "TC": [], "ET": []}

    print("Processing patients")

    limit = int(os.environ.get("EVAL_LIMIT", 0))
    patient_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            if limit > 0 and patient_count >= limit:
                break
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

            # each element in the batch (batch_size is 1 at validation)
            for logit, label in zip(decollate_batch(logits), decollate_batch(labels)):
                if limit > 0 and patient_count >= limit:
                    break
                patient_count += 1
                prob = torch.softmax(logit, dim=0)

                region_probs = convert_to_region_probs(prob)
                region_preds_bin = convert_to_region_binary(prob)
                region_labels = convert_to_regions(label)

                dice_metric(y_pred=region_preds_bin.unsqueeze(0), y=region_labels.unsqueeze(0))
                hd95_metric(y_pred=region_preds_bin.unsqueeze(0), y=region_labels.unsqueeze(0))

                for idx, region_name in enumerate(["WT", "TC", "ET"]):
                    pred_mask = region_preds_bin[idx] > 0.5
                    true_mask = region_labels[idx] > 0.5

                    pred_prob_np = region_probs[idx].cpu().numpy().flatten()
                    true_mask_np = true_mask.cpu().numpy().flatten()
                    pred_mask_np = pred_mask.cpu().numpy().flatten()

                    tp = np.sum(pred_mask_np & true_mask_np)
                    fp = np.sum(pred_mask_np & ~true_mask_np)
                    fn = np.sum(~pred_mask_np & true_mask_np)

                    if (tp + fn) > 0:
                        rec = tp / (tp + fn)
                    else:
                        rec = float('nan')

                    if (tp + fp) > 0:
                        prec = tp / (tp + fp)
                    else:
                        prec = float('nan')

                    if np.sum(true_mask_np) > 0:
                        try:
                            auc_pr = average_precision_score(true_mask_np, pred_prob_np)
                        except Exception as e:
                            print(f"Error computing PR AUC for {region_name}: {e}")
                            auc_pr = float('nan')
                    else:
                        auc_pr = float('nan')

                    precision_lists[region_name].append(prec)
                    recall_lists[region_name].append(rec)
                    pr_auc_lists[region_name].append(auc_pr)

    raw_dice = dice_metric.aggregate()
    if isinstance(raw_dice, (tuple, list)):
        raw_dice = raw_dice[0]

    raw_hd = hd95_metric.aggregate()
    if isinstance(raw_hd, (tuple, list)):
        raw_hd = raw_hd[0]

    print("\n" + "=" * 50)
    print("FINAL RESULTS (DICE, HD95, PRECISION, RECALL, PR AUC):")
    print("=" * 50)

    for idx, region_name in enumerate(["WT", "TC", "ET"]):
        r_dice = raw_dice[:, idx]
        dice_m = r_dice[~torch.isnan(r_dice)].mean().item()
        valid_dice_count = len(r_dice) - torch.isnan(r_dice).sum().item()

        r_hd = raw_hd[:, idx]
        valid_hd = r_hd[~torch.isnan(r_hd) & ~torch.isinf(r_hd)]
        hd_m = valid_hd.mean().item() if len(valid_hd) > 0 else float('nan')
        valid_hd_count = len(valid_hd)

        p_list = precision_lists[region_name]
        p_tensor = torch.tensor(p_list)
        prec_m = p_tensor[~torch.isnan(p_tensor)].mean().item()
        valid_p_count = len(p_list) - torch.isnan(p_tensor).sum().item()

        r_list = recall_lists[region_name]
        r_tensor = torch.tensor(r_list)
        rec_m = r_tensor[~torch.isnan(r_tensor)].mean().item()
        valid_r_count = len(r_list) - torch.isnan(r_tensor).sum().item()

        auc_list = pr_auc_lists[region_name]
        auc_tensor = torch.tensor(auc_list)
        auc_m = auc_tensor[~torch.isnan(auc_tensor)].mean().item()
        valid_auc_count = len(auc_list) - torch.isnan(auc_tensor).sum().item()

        print(f"Region: {region_name}")
        print(f"  Dice (DSC): {dice_m:.4f} (n={int(valid_dice_count)} patients)")
        print(f"  HD95:       {hd_m:.4f} (n={int(valid_hd_count)} patients)")
        print(f"  Precision:  {prec_m:.4f} (n={int(valid_p_count)} patients)")
        print(f"  Recall:     {rec_m:.4f} (n={int(valid_r_count)} patients)")
        print(f"  PR AUC:     {auc_m:.4f} (n={int(valid_auc_count)} patients)")
        print("-" * 50)

    print("=" * 50)


if __name__ == "__main__":
    main()
