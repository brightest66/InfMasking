from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from evaluation.linear_probe import LinearProbingCallback

# pass the current epoch and total epochs to the model
class EpochInfoCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        pl_module.set_current_epoch(current_epoch)

class TotalEpochsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_epochs = trainer.max_epochs
        pl_module.set_total_epochs(total_epochs)

@hydra.main(version_base=None, config_name="train_trifeatures", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on synthetic toy data (bimodal trifeatures) with
    controllable attributes (shape, color, texture).

    Models currently implemented are:
        - CLIP
        - CrossSelf
        - CoMM
        - InfMasking
    """

    # fix the seed for repro
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # create model + save hyper-parameters
    kwargs = dict()

    if cfg.model.name== "CoMM" or cfg.model.name == "InfMasking":
        kwargs["encoder"] = {
            "encoders": instantiate(cfg.model.encoders),
            "input_adapters": instantiate(cfg.model.adapters)}

    if cfg.model.name == "CLIP":
        encoders = instantiate(cfg.model.encoders)
        kwargs["visual"], kwargs["language"] = encoders[0], encoders[1]
        kwargs["image_projection"] = instantiate(cfg.model.clip_image_projection)
        kwargs["text_projection"] = instantiate(cfg.model.clip_text_projection)

    if cfg.model.name == "CrossSelf":
        encoders = instantiate(cfg.model.encoders)
        kwargs["enc1"] = encoders[0]
        kwargs["enc2"] = encoders[1]
        kwargs["head1"] = instantiate(cfg.model.visual_projection)
        kwargs["head2"] = instantiate(cfg.model.visual_projection)


    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)

    model.save_hyperparameters(cfg)

    # Data loading code
    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    # Linear probing on each tasks from BimodalTrifeatures
    downstream_names = ["share", "unique1", "unique2", "synergy"]
    downstream_data_modules = [instantiate(cfg.data.data_module, model="Sup", biased=False, task=t)
                               for t in downstream_names]
    logger = TensorBoardLogger(build_root_dir(cfg), name="logs")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        # dirpath=build_root_dir(cfg),  
        dirpath=os.path.join(logger.log_dir, "checkpoints"), 
        filename='{epoch:02d}-{acc1:.4f}', 
        save_top_k=1,  
        monitor='acc1',
        mode='max',  
    )       

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='acc1', 
        patience=10,  
        verbose=True, 
        mode='max' 
    )                   
    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=[TensorBoardLogger(build_root_dir(cfg), name="logs")],
        callbacks=[
            EpochInfoCallback(),
            TotalEpochsCallback(),
            LinearProbingCallback(downstream_data_modules,
                                         names=downstream_names,
                                         val_loaders=False), # if False, split the val from the train set
                                         early_stopping_callback,
                                         checkpoint_callback],
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=getattr(cfg, "ckpt_path", None))


def build_root_dir(cfg: DictConfig):
    # set directory for logs and checkpoints
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "bimodal_trifeatures")

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