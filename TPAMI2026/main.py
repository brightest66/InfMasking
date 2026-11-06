import importlib
import inspect
import json
import os
import time
from math import inf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train_func import train
from utils.args import get_arguments
from utils.utils import *
from validation import valid
from models.fusion_modules import MultimodalFusionBase
from models.input_adapters import FeaturesInputAdapter, PatchedInputAdapter
from models.mm_transformer import MT2Model, MTModel
from models.unimodal_encoders import (
    BertEncoder,
    GRU,
    MLP,
    Transformer,
    TvFeatureExtractor,
)
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DATA_FILES = [
    os.path.join(PROJECT_ROOT, "configs", "data", n)
    for n in ("multibench.json", "multibenchplus.json")
]
CONFIG_MODEL_FILES = [
    os.path.join(PROJECT_ROOT, "configs", "models", n) for n in ("regular_models.json",)
]
CLASS_REGISTRY = {
    cls.__name__: cls
    for cls in (
        Transformer,
        GRU,
        MLP,
        BertEncoder,
        TvFeatureExtractor,
        PatchedInputAdapter,
        FeaturesInputAdapter,
    )
}
dim_mapping = {
    "text": "t_feature_dim",
    "vision": "v_feature_dim",
    "audio": "a_feature_dim",
    "tabular": "t_feature_dim",
    "atimeseries": "a_feature_dim",
}


def get_loader(args):
    data_registry = load_data_registry(CONFIG_DATA_FILES)
    spec = (data_registry.get("datasets") or {}).get(args.dataset, {}) or {}
    args.modalities = (
        args.modalities or spec.get("default_modalities") or ["text", "vision"]
    )
    set_arg_unless_cli_overridden(args, "batch_size", spec.get("batch_size"))
    fold = int(getattr(args, "fold", 0))
    data_split = spec.get("data_split")
    if isinstance(data_split, dict):
        data_split["fold"] = fold
    elif data_split is None:
        spec["data_split"] = {"fold": fold}
    supported = spec.get("modalities_supported")
    if supported is not None:
        unsupported = [m for m in args.modalities if m not in supported]
        if unsupported:
            raise ValueError(
                f"Dataset '{args.dataset}' does not support modalities {unsupported}. Supported: {supported}"
            )
    set_args = spec.get("set_args", {})
    edim = (
        args.multibench_dim
        if set_args.get("edim_from") == "multibench_dim"
        else set_args.get("edim")
    )
    set_arg_unless_cli_overridden(args, "edim", int(edim) if edim is not None else None)
    feat_dims = spec.get("feature_dims", {}) or {}
    for modality, attr in dim_mapping.items():
        if modality in feat_dims:
            setattr(args, attr, int(feat_dims[modality]))
    loader_spec = spec.get("loader") or data_registry.get("default_loader")
    if not loader_spec:
        raise ValueError(
            f"No dataloader config found for dataset '{args.dataset}'. Please fill configs/data/*.json."
        )
    loader_fn = getattr(
        importlib.import_module(loader_spec["module"]), loader_spec["callable"]
    )
    extra = {}
    for loader_kw, arg_attr in (spec.get("loader_kwargs_from_args") or {}).items():
        extra[loader_kw] = getattr(args, arg_attr)
    train_dataloader, test_dataloader, valid_dataloader, n_classes = loader_fn(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        modalities=args.modalities,
        **extra,
    )
    maybe_infer_feature_dims(args, train_dataloader)
    return (n_classes, train_dataloader, test_dataloader, valid_dataloader, spec)


def build_model(args, n_classes, device, model_registry: dict):
    dataset_specs = model_registry.get("dataset_specs") or {}
    ds = dataset_specs.get(args.dataset)
    if ds is None:
        raise KeyError(
            f"Missing model spec for dataset '{args.dataset}' in configs/models/regular_models.json"
        )
    freeze = args.fusion_method in (ds.get("freeze_when_fusion_methods") or [])
    context = {**vars(args), "freeze": freeze}
    encoders_config = ds.get("encoders") or {}
    for name, spec in encoders_config.items():
        kwargs = spec.get("kwargs")
        if kwargs is None:
            kwargs = {}
            spec["kwargs"] = kwargs
        cls_name = spec.get("class")
        cls = CLASS_REGISTRY.get(cls_name)
        if cls is None:
            continue
        init_signature = inspect.signature(cls.__init__)
        accepts_var_kwargs = any(
            (
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in init_signature.parameters.values()
            )
        )
        if "args" in init_signature.parameters or accepts_var_kwargs:
            kwargs.setdefault("args", args)
    encoders = {
        k: instantiate_from_spec(v, context, CLASS_REGISTRY)
        for k, v in encoders_config.items()
    }
    input_adapters = {
        k: instantiate_from_spec(v, context, CLASS_REGISTRY)
        for k, v in (ds.get("input_adapters") or {}).items()
    }
    pooling_cfg = ds.get("pooling") or {}
    if pooling_cfg:
        pooling = {m: pooling_cfg.get(m, "last") for m in args.modalities}
    fusion = {"MT": MTModel, "MT2": MT2Model}[args.fusion_method](
        args=args, n_classes=n_classes, pooling=pooling
    )
    use_enc = True
    model = MultimodalFusionBase(
        args=args,
        encoders=encoders,
        input_adapters=input_adapters,
        fusion_module=fusion,
        use_enc=use_enc,
    )
    model.fusion.apply(weight_init)
    for key in model.input_adapters:
        if model.input_adapters[key] is not None:
            model.input_adapters[key].apply(weight_init)
    model.to(device)
    return model


def main():
    start_time = time.time()
    args = get_arguments()
    setup_seed(args.random_seed)
    device = torch.device(args.device)
    model_registry = load_model_registry(CONFIG_MODEL_FILES)
    n_classes, train_dataloader, test_dataloader, valid_dataloader, dataset_spec = (
        get_loader(args=args)
    )
    args.n_classes = n_classes
    train_defaults = dataset_spec.get("train_defaults", {}) or {}
    apply_train_defaults(args, train_defaults)
    args.epochs = int(args.epochs)
    auto_select_fusion(args, model_registry)
    apply_model_arg_overrides(args, model_registry)
    model = build_model(
        args=args, n_classes=n_classes, device=device, model_registry=model_registry
    )
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-08,
            weight_decay=args.weight_decay,
        )
    total_steps = len(train_dataloader) * args.epochs
    if args.scheduler == "cosine":
        warmup_steps = (
            args.warmup_steps
            if args.warmup_steps >= 0
            else int(total_steps * args.warmup_ratio)
        )
        warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
        args.warmup_steps = warmup_steps
        t0_epochs = max(1, int(getattr(args, "cosine_t0_epochs", 4)))
        t_mult = max(1, int(getattr(args, "cosine_t_mult", 2)))
        t0_steps = max(1, int(len(train_dataloader) * t0_epochs))
        after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0_steps, T_mult=t_mult
        )
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            base_lr=args.learning_rate,
            n_warmup_steps=warmup_steps,
            after_scheduler=after_scheduler,
        )
    else:
        scheduler = None
    print(f"Use {args.optimizer} optimizer and {args.scheduler} scheduler.")
    exp_root, ckpt_path = prepare_exp(args=args)
    save_run_command(exp_root)
    train_logger, val_logger = get_logger(exp_root)
    with open(os.path.join(exp_root, "config.json"), "w", encoding="utf-8") as file:
        file.write(json.dumps(args_to_json_dict(args), separators=(",\n", ":")))
    print(
        "CORE TRAIN ARGS:",
        f"lr: {args.learning_rate}, bs: {args.batch_size}, epoch: {args.epochs}",
    )
    best_acc = -inf
    best_epoch = -1
    best_log = os.path.join(ckpt_path, "best.txt")
    best_ckpt_path = None
    no_improve = 0
    writer = (
        None
        if getattr(args, "tb_disable", False)
        else SummaryWriter(os.path.join(exp_root, "log"))
    )
    tb_log_interval = int(getattr(args, "tb_log_interval", 50))
    global_step = 0
    for epoch in range(
        min(args.epochs, args.stop_after_epochs)
        if args.stop_after_epochs > 0
        else args.epochs
    ):
        batch_loss, batch_loss_a, batch_loss_v, batch_loss_mask = train(
            args,
            epoch,
            model,
            device,
            train_dataloader,
            optimizer,
            scheduler,
            train_logger,
            writer=writer,
            global_step_start=global_step,
            log_interval=tb_log_interval,
        )
        if writer is not None:
            writer.add_scalars(
                "Loss",
                {
                    "Total Loss": batch_loss,
                    "Audio Loss": batch_loss_a,
                    "Visual Loss": batch_loss_v,
                    "Masking Loss": batch_loss_mask,
                },
                epoch,
            )
        current_lr = float(optimizer.param_groups[-1]["lr"])
        if writer is not None:
            writer.add_scalar("train/lr_epoch", current_lr, epoch)
        acc = valid(
            args,
            model,
            device,
            valid_dataloader,
            epoch,
            val_logger,
            name="Val",
            writer=writer,
            global_step=epoch,
        )
        if writer is not None:
            writer.add_scalars("Evaluation", {"Total Accuracy": acc}, epoch)
        print("Loss: {:.3f}, Acc: {:.5f}".format(batch_loss, acc))
        print("Previous Best Acc: {:.5f}".format(best_acc))
        improved = float(acc) >= float(best_acc) + float(args.early_stop_min_delta)
        if improved:
            best_acc = float(acc)
            best_epoch = int(epoch)
            with open(best_log, "w", encoding="utf-8") as file:
                file.write("epoch {}, acc {:.5f}".format(best_epoch, best_acc))
            saved_dict = {
                "saved_epoch": best_epoch,
                "fusion": args.fusion_method,
                "acc": acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            best_ckpt_path = os.path.join(ckpt_path, "best.pth")
            torch.save(saved_dict, best_ckpt_path)
            print("The best model has been saved at {}.".format(best_ckpt_path))
            no_improve = 0
        else:
            no_improve += 1
            min_epochs = max(0, int(getattr(args, "early_stop_min_epochs", 0)))
            if (
                args.early_stop_patience > 0
                and no_improve >= args.early_stop_patience
                and (epoch + 1 >= min_epochs)
            ):
                print(
                    f"Early stopping triggered (patience={args.early_stop_patience}). Best epoch={best_epoch}, best acc={best_acc:.5f}"
                )
                break
        global_step += len(train_dataloader)
    end_time = time.time()
    total_time_trained = end_time - start_time
    print(f"Loading the best model from {ckpt_path} for Testing...")
    start_time = time.time()
    if best_ckpt_path is None:
        raise RuntimeError(
            "No checkpoint was saved. Check validation loop / dataloader."
        )
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    test_acc = valid(
        args,
        model,
        device,
        test_dataloader,
        ckpt.get("saved_epoch", -1),
        val_logger,
        name="Test",
        writer=writer,
        global_step=ckpt.get("saved_epoch", -1),
    )
    end_time = time.time()
    total_time_tested = end_time - start_time
    if writer is not None:
        writer.close()
    print("Test Acc: {:.5f} ".format(test_acc))
    with open(best_log, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write(
            "Model loaded from {}.\nTest Acc (best epoch {}): {:.5f}".format(
                best_ckpt_path, ckpt.get("saved_epoch", -1), test_acc
            )
        )
        file.write("\n")
        file.write(f"Total time trained: {total_time_trained:.2f} seconds\n")
        file.write(f"Total time tested: {total_time_tested:.2f} seconds\n")


if __name__ == "__main__":
    main()
