import torch
import sys
import os
from pathlib import Path
import typing
import numpy as np
from scipy.ndimage import label

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.models.components.multi_task_swin import MultiTaskSwinUNETR
from src.models.brats_module import BraTSLightningModule
from monai.inferers import sliding_window_inference
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


def evaluate_lesion_wise(y_pred: np.ndarray, y_true: np.ndarray):
    pred_labeled, num_preds = label(y_pred)
    true_labeled, num_trues = label(y_true)

    recall = None
    precision = None

    if num_trues > 0:
        true_detected = 0
        for t in range(1, num_trues + 1):
            true_mask = (true_labeled == t)
            if np.sum(y_pred[true_mask]) > 0:
                true_detected += 1
        recall = true_detected / num_trues

    if num_preds > 0:
        false_positives = 0
        for p in range(1, num_preds + 1):
            pred_mask = (pred_labeled == p)
            if np.sum(y_true[pred_mask]) == 0:
                false_positives += 1
        precision = (num_preds - false_positives) / num_preds
    elif num_trues == 0:
        precision = 1.0
    else:
        precision = 0.0

    return precision, recall


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    CHECKPOINT_PATH = os.environ.get(
        "CHECKPOINT_PATH",
        "/work/ab0995/a270263/mgr/logs/brats-mgr-project/unwx02xp/checkpoints/resume.ckpt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running lesion-wise evaluation on {device}")

    print("Initializing dataloader")
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    print("Loading model")
    net = MultiTaskSwinUNETR(in_channels=4, num_classes=4, feature_size=48)
    model = BraTSLightningModule.load_from_checkpoint(CHECKPOINT_PATH, net=net, map_location=device)
    model.to(device)
    model.eval()

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

            for p_pred, p_true in zip(region_preds, region_labels):
                p_pred_np = p_pred.cpu().numpy()
                p_true_np = p_true.cpu().numpy()

                for idx, region_name in enumerate(["WT", "TC", "ET"]):
                    precision, recall = evaluate_lesion_wise(p_pred_np[idx], p_true_np[idx])

                    if precision is not None:
                        precision_lists[region_name].append(precision)
                    if recall is not None:
                        recall_lists[region_name].append(recall)

    print("\n" + "=" * 40)
    print("FINAL RESULTS, LESION-WISE (DETECTION):")
    print("=" * 40)

    mean_f1_sum = 0.0

    for region_name in ["WT", "TC", "ET"]:
        p_list = precision_lists[region_name]
        r_list = recall_lists[region_name]

        mean_p = np.mean(p_list) if len(p_list) > 0 else 0.0
        mean_r = np.mean(r_list) if len(r_list) > 0 else 0.0

        if mean_p + mean_r > 0:
            f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r)
        else:
            f1 = 0.0

        mean_f1_sum += f1

        print(f"Region: {region_name}")
        print(f"  Precision: {mean_p:.4f} (n={len(p_list)} patients)")
        print(f"  Recall:    {mean_r:.4f} (n={len(r_list)} patients)")
        print(f"  F1-Score:  {f1:.4f}")
        print("-" * 40)

    print(f"MEAN BraTS F1 (lesion-wise): {mean_f1_sum / 3:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
