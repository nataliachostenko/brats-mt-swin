import os
import sys
from pathlib import Path
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, SaveImage
from monai.inferers import sliding_window_inference

project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.brats_module import BraTSLightningModule
from src.models.components.multi_task_swin import MultiTaskSwinUNETR

import typing
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


def predict_patient(checkpoint_path: str, patient_folder: str, output_dir: str):
    print(f"Running inference for patient: {patient_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net = MultiTaskSwinUNETR(in_channels=4, num_classes=4, feature_size=48)
    model = BraTSLightningModule.load_from_checkpoint(checkpoint_path, net=net)
    model.eval()
    model.to(device)

    t1n = os.path.join(patient_folder, [f for f in os.listdir(patient_folder) if f.endswith('t1n.nii.gz')][0])
    t1c = os.path.join(patient_folder, [f for f in os.listdir(patient_folder) if f.endswith('t1c.nii.gz')][0])
    t2w = os.path.join(patient_folder, [f for f in os.listdir(patient_folder) if f.endswith('t2w.nii.gz')][0])
    t2f = os.path.join(patient_folder, [f for f in os.listdir(patient_folder) if f.endswith('t2f.nii.gz')][0])

    data_dict = {"image": [t1n, t1c, t2w, t2f]}

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    ])

    input_data = transforms(data_dict)
    image_tensor = input_data["image"].unsqueeze(0).to(device)

    print("Running sliding-window inference")
    with torch.no_grad():
        def predictor_wrapper(x):
            out = model.net(x)
            return out[0] if isinstance(out, tuple) else out

        val_outputs = sliding_window_inference(
            inputs=image_tensor,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=predictor_wrapper,
            overlap=0.5
        )

        pred_mask = torch.argmax(val_outputs, dim=1)

    print(f"Saving predicted mask to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    pred_mask = pred_mask.cpu()
    original_meta_dict = input_data["image"].meta

    saver = SaveImage(
        output_dir=output_dir,
        output_postfix="predicted_mask",
        output_ext=".nii.gz",
        resample=False,
        separate_folder=False
    )

    saver(pred_mask, meta_data=original_meta_dict)
    print("File saved")


if __name__ == "__main__":
    CHECKPOINT = "/work/ab0995/a270263/logs/brats-mgr-project/r7uperj0/checkpoints/resume.ckpt"
    PATIENT = "/work/ab0995/a270263/mgr/data/BraTS-GLI/extracted_data/validation_data/BraTS-GLI-02073-100"
    OUTPUT = "./predictions/"

    predict_patient(CHECKPOINT, PATIENT, OUTPUT)
