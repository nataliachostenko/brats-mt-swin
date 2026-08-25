import hydra
from omegaconf import DictConfig
import torch
import omegaconf
from hydra.utils import instantiate

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig])

# newer torch defaults torch.load to weights_only=True, which breaks loading
# checkpoints that pickle OmegaConf objects
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
torch.set_float32_matmul_precision('medium')


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.task_name}")

    print("Building dataloader")
    datamodule = instantiate(cfg.data)

    print("Building model")
    model = instantiate(cfg.model)

    print("Connecting to W&B")
    logger = instantiate(cfg.logger)

    print("Configuring trainer")
    trainer = instantiate(cfg.trainer, logger=logger)

    print("Starting training — check the W&B dashboard for progress")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
