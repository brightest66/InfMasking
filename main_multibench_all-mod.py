from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

# pass the current epoch and total epochs to the model
class EpochInfoCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        pl_module.set_current_epoch(current_epoch)

class TotalEpochsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_epochs = trainer.max_epochs
        pl_module.set_total_epochs(total_epochs)

@hydra.main(version_base=None, config_name="train_multibench_all-mod", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on MultiBench dataset.
    Models currently implemented are:
        - CoMM
        - InfMasking
    """
    # fix the seed for repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # create model + save hyper-parameters
    dataset = cfg.data.data_module.dataset # Which MultiBench dataset to load
    kwargs = dict()

    encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
    adapters = instantiate(cfg[dataset]["adapters"]) # adapters also specific
    kwargs["encoder"] = {
        "encoders": encoders,
        "input_adapters": adapters}

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module,
                              model=cfg.model.name,
                              modalities=cfg[dataset]["modalities"],
                              task=cfg[dataset]["task"],
                              **cfg[dataset]["kwargs"])

    downstream_data_module = instantiate(cfg.data.data_module,
                                         model="Sup",
                                         modalities=cfg[dataset]["modalities"],
                                         task=cfg[dataset]["task"])
    logger = TensorBoardLogger(build_root_dir(cfg), name="logs")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch:02d}-{acc1:.4f}",
        monitor="acc1",
        mode="max",
        save_top_k=1,
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="acc1",
        patience=20,
        verbose=True,
        mode="max",
    )

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir = build_root_dir(cfg),
        logger=[logger],
        callbacks=[
            instantiate(cfg.linear_probing,downstream_data_modules=[downstream_data_module],names=[dataset]),
                                checkpoint_callback,
                                early_stop_callback,
                                EpochInfoCallback(),
                                TotalEpochsCallback()
                               ],
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, cfg.data.data_module.dataset)

    # modify `root_dir` if in test mode to match pre-trained model's path
    if cfg.mode == "test":
        if cfg.ckpt_path is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == '__main__':
    main()