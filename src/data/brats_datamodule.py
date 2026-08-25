import os
import glob
from typing import Optional

import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    MapLabelValued,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)


class BraTSDataModule(pl.LightningDataModule):
    """Lightning DataModule for the BraTS 2024 GLI dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        roi_size: list = [96, 96, 96],
        num_samples_per_image: int = 4,
        seed: int = 42,
        val_split: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_data = []
        self.val_data = []

        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()

    def _get_train_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            # brain-only z-score: background voxels would otherwise skew mean/std
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # BraTS labels are 0,1,2,4 (historical gap) — remap 4 to 3
            MapLabelValued(keys=["seg"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
            # pos=1, neg=1: half the patches are centered on tumor, half on background
            RandCropByPosNegLabeld(
                keys=["image", "seg"],
                label_key="seg",
                spatial_size=self.hparams.roi_size,
                pos=1.0,
                neg=1.0,
                num_samples=self.hparams.num_samples_per_image,
                image_key="image",
                image_threshold=0.0
            ),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ])

    def _get_val_transforms(self):
        # no cropping — the full volume is handled by sliding-window inference at eval time
        return Compose([
            LoadImaged(keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MapLabelValued(keys=["seg"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3])
        ])

    def setup(self, stage: Optional[str] = None):
        mask_files = sorted(glob.glob(os.path.join(self.hparams.data_dir, "**", "*-seg.nii.gz"), recursive=True))

        data_dicts = []
        for mask_path in mask_files:
            patient_dir = os.path.dirname(mask_path)

            t1n_path = glob.glob(os.path.join(patient_dir, "*-t1n.nii.gz"))
            t1c_path = glob.glob(os.path.join(patient_dir, "*-t1c.nii.gz"))
            t2w_path = glob.glob(os.path.join(patient_dir, "*-t2w.nii.gz"))
            t2f_path = glob.glob(os.path.join(patient_dir, "*-t2f.nii.gz"))

            if all([t1n_path, t1c_path, t2w_path, t2f_path]):
                data_dicts.append({
                    "image": [t1n_path[0], t1c_path[0], t2w_path[0], t2f_path[0]],
                    "seg": mask_path
                })

        print(f"Found {len(data_dicts)} complete patients. Splitting...")

        train_files, val_files = train_test_split(
            data_dicts,
            test_size=self.hparams.val_split,
            random_state=self.hparams.seed
        )

        self.train_data = Dataset(data=train_files, transform=self.train_transforms)
        self.val_data = Dataset(data=val_files, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        # a full volume per sample, so batch_size is fixed at 1
        return DataLoader(
            self.val_data,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )
