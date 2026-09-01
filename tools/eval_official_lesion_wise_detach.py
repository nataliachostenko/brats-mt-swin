import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import typing
import omegaconf
import omegaconf.base

torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig, 
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any
])

import functools
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

metrics_dir = os.path.join(project_root, "tools", "BraTS2024Metrics")
if metrics_dir not in sys.path:
    sys.path.insert(0, metrics_dir)

from src.models.components.multi_task_swin_detach import MultiTaskSwinUNETRDetach
from src.models.brats_module import BraTSLightningModule
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch

from metrics_GLI import get_LesionWiseResults

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    CHECKPOINT_PATH = os.environ.get(
        "CHECKPOINT_PATH", 
        "/work/ab0995/a270263/mgr/logs/brats-mgr-project/jnopa6kt/checkpoints/epoch=99-step=16200.ckpt"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Started official lesion-wise evaluation on {device}")
    print(f"Using checkpoint: {CHECKPOINT_PATH}")

    POSTPROCESS_MIN_VOXELS = int(os.environ.get("POSTPROCESS_MIN_VOXELS", 0))
    if POSTPROCESS_MIN_VOXELS > 0:
        print(f"Post-processing enabled. Removing components < {POSTPROCESS_MIN_VOXELS} voxels")

    scratch_dir = os.path.join(project_root, "scratch", "temp_eval")
    os.makedirs(scratch_dir, exist_ok=True)

    print("Initializing dataloader")
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()

    print("Loading model")
    net = MultiTaskSwinUNETRDetach(in_channels=4, num_classes=4, feature_size=48)
    model = BraTSLightningModule.load_from_checkpoint(CHECKPOINT_PATH, net=net, map_location=device)
    model.to(device)
    model.eval()

    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    patient_dfs = []
    patient_count = 0
    limit = int(os.environ.get("EVAL_LIMIT", 0))
    if limit > 0:
        print(f"Limiting evaluation to first {limit} patients")

    print("Processing patients...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if limit > 0 and patient_count >= limit:
                break
            images = batch["image"].to(device)
            labels = batch["seg"].to(device)
            
            gt_filepath = batch["seg"].meta["filename_or_obj"][0] if hasattr(batch["seg"], "meta") else batch["seg_meta_dict"]["filename_or_obj"][0]
            patient_id = Path(gt_filepath).parent.name
            patient_count += 1

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
            pred_mask = preds[0] # batch_size is 1 in val

            pred_np = pred_mask.squeeze().cpu().numpy().astype(np.int32)
            gt_np = labels.squeeze().cpu().numpy().astype(np.int32)

            if POSTPROCESS_MIN_VOXELS > 0:
                from src.utils.postprocess import remove_small_islands
                for c in [1, 2, 3]:
                    class_mask = (pred_np == c)
                    filtered_class_mask = remove_small_islands(class_mask, POSTPROCESS_MIN_VOXELS)
                    pred_np[(class_mask) & (~filtered_class_mask)] = 0

            if hasattr(batch["seg"], "affine"):
                affine = batch["seg"].affine[0].cpu().numpy()
            elif "seg_meta_dict" in batch and "affine" in batch["seg_meta_dict"]:
                affine = batch["seg_meta_dict"]["affine"][0].cpu().numpy()
            else:
                affine = np.eye(4)

            temp_pred_path = os.path.join(scratch_dir, f"{patient_id}_pred.nii.gz")
            temp_gt_path = os.path.join(scratch_dir, f"{patient_id}_gt.nii.gz")

            pred_nii = nib.Nifti1Image(pred_np, affine)
            gt_nii = nib.Nifti1Image(gt_np, affine)
            nib.save(pred_nii, temp_pred_path)
            nib.save(gt_nii, temp_gt_path)

            try:
                results_df, _ = get_LesionWiseResults(
                    pred_file=temp_pred_path,
                    gt_file=temp_gt_path,
                    challenge_name="BraTS-GLI",
                    output=None
                )
                
                results_df["patient_id"] = patient_id
                patient_dfs.append(results_df)

            except Exception as e:
                print(f"\nError evaluating patient {patient_id}: {e}")

            finally:
                if os.path.exists(temp_pred_path):
                    os.remove(temp_pred_path)
                if os.path.exists(temp_gt_path):
                    os.remove(temp_gt_path)
                
                # metrics_GLI.py leaves these behind as a side effect
                if os.path.exists("./tmp_gt"):
                    shutil.rmtree("./tmp_gt")
                if os.path.exists("./tmp_pred"):
                    shutil.rmtree("./tmp_pred")

    if len(patient_dfs) > 0:
        all_results = pd.concat(patient_dfs, ignore_index=True)
        
        suffix = ""
        if limit > 0:
            suffix += f"_limit_{limit}"
        if POSTPROCESS_MIN_VOXELS > 0:
            suffix += f"_pp_{POSTPROCESS_MIN_VOXELS}"
            
        raw_filename = f"official_lesionwise_detach{suffix}_raw.csv"
        summary_filename = f"official_lesionwise_detach{suffix}_summary.csv"
        
        raw_output_path = os.path.join(project_root, "outputs", raw_filename)
        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        all_results.to_csv(raw_output_path, index=False)
        print(f"\nRaw results saved to {raw_output_path}")

        print("\n" + "="*50)
        print("FINAL OFFICIAL LESION-WISE METRICS (BraTS 2024 GLI):")
        print("="*50)
        
        brats_regions = ["WT", "TC", "ET"]
        
        summary_rows = []
        for region in brats_regions:
            region_df = all_results[all_results["Labels"] == region]
            if len(region_df) > 0:
                mean_ldice = region_df["LesionWise_Score_Dice"].mean()
                mean_lhd95 = region_df["LesionWise_Score_HD95"].mean()
                mean_lnsd05 = region_df["LesionWise_Score_NSD @ 0.5"].mean()
                mean_lnsd10 = region_df["LesionWise_Score_NSD @ 1.0"].mean()
                
                print(f"Region: {region}")
                print(f"  Lesion-wise Dice (L-Dice):        {mean_ldice:.4f}")
                print(f"  Lesion-wise HD95 (L-HD95):        {mean_lhd95:.4f}")
                print(f"  Lesion-wise NSD @ 0.5:           {mean_lnsd05:.4f}")
                print(f"  Lesion-wise NSD @ 1.0:           {mean_lnsd10:.4f}")
                print("-" * 50)
                
                summary_rows.append({
                    "Region": region,
                    "L-Dice": mean_ldice,
                    "L-HD95": mean_lhd95,
                    "L-NSD@0.5": mean_lnsd05,
                    "L-NSD@1.0": mean_lnsd10
                })
        
        if len(summary_rows) > 0:
            summary_df = pd.DataFrame(summary_rows)
            summary_output_path = os.path.join(project_root, "outputs", summary_filename)
            summary_df.to_csv(summary_output_path, index=False)
            print(f"Summary saved to {summary_output_path}")
            
            avg_ldice = summary_df["L-Dice"].mean()
            avg_lhd95 = summary_df["L-HD95"].mean()
            print(f"OVERALL AVERAGE:")
            print(f"  Mean L-Dice:                      {avg_ldice:.4f}")
            print(f"  Mean L-HD95:                      {avg_lhd95:.4f}")
            
        print("="*50)
    else:
        print("No evaluation results were computed successfully.")

if __name__ == "__main__":
    main()
