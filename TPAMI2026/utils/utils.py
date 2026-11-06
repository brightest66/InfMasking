import torch
import torch.nn as nn
import numpy as np
import random
import os
import csv
import time
import json
import sys
import shlex


def prepare_exp(args):
    # setup exp
    # modalities can be None before dataset config is applied
    if getattr(args, "modalities", None):
        modalities_acronym = "".join([m[0].upper() for m in args.modalities])
    else:
        modalities_acronym = "UNK"
    if args.use_masking:
        objective_dir = "InfMasking"
        experiment_name = objective_dir + "/"
    else:
        experiment_name = ""
    if args.exp_name == "":
        # experiment_name = args.fusion_method + "-seed_" + str(args.random_seed) + "-" + modalities_acronym
        experiment_name += (
            args.fusion_method
            + "-"
            + modalities_acronym
            + "/"
            + "seed_"
            + str(args.random_seed)
        )
    else:
        experiment_name += (
            args.exp_name
            + "-"
            + modalities_acronym
            + "/"
            + "seed_"
            + str(args.random_seed)
        )
    if args.fold_use:
        experiment_name += "-fold_" + str(args.fold)
    experiment = experiment_name + "-" + time.strftime("%y%m%d_%H:%M", time.localtime())

    sub_root = "{}".format(args.dataset)
    print("NAME OF EXPERIMENT: {}_{}".format(sub_root, experiment))

    # ``--exp_root`` is the optional base directory supplied by the caller.
    # Keep the historical ``exp/<dataset>`` layout when it is omitted.
    exp_base_root = getattr(args, "exp_root", "") or "exp"
    exp_dset_root = os.path.join(exp_base_root, sub_root)
    if not os.path.exists(exp_dset_root):
        os.makedirs(exp_dset_root)

    exp_root = os.path.join(exp_dset_root, experiment)
    args.exp_root = exp_root
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    ckpt_path = os.path.join(exp_root, "ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    return exp_root, ckpt_path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_logger(path):
    train_logger = Logger(os.path.join(path, "train.log"), ["epoch", "loss", "lr"])

    val_logger = Logger(os.path.join(path, "val.log"), ["epoch", "loss", "acc"])
    return train_logger, val_logger


def get_pooling(pooling):
    if pooling == "last":
        return lambda x: x[:, -1, :]
    elif pooling == "first":
        return lambda x: x[:, 0, :]
    elif pooling == "second":
        return lambda x: x[1]
    elif pooling == "mean":
        return lambda x: x.mean(dim=1)
    else:
        raise ValueError(f"Invalid pooling method: {pooling}")


class Logger(object):
    def __init__(self, path, header):
        # self.log_file = path.open('w')
        self.log_file = open(path, "w")
        self.logger = csv.writer(self.log_file, delimiter="\t")

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def _move_to_device(obj, device):
    """Recursively move tensors (and dicts/lists/tuples of tensors) to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_move_to_device(v, device) for v in obj]
    return obj


class WarmupScheduler(nn.Module):
    """
    Per-step warmup wrapper around a PyTorch LR scheduler.

    - During warmup (first n_warmup_steps steps): linearly ramp lr from 0 to `base_lr`.
    - After warmup: delegate to `after_scheduler.step(step - n_warmup_steps)` and track lr history.

    Notes:
    - Designed to be called every training step (i.e., per batch).
    - Works with existing training code: `if scheduler is not None: scheduler.step()`.
    """

    def __init__(self, optimizer, base_lr, n_warmup_steps, after_scheduler):
        super().__init__()
        self._optimizer = optimizer
        self.base_lr = float(base_lr)
        self.n_warmup_steps = int(n_warmup_steps)
        self.n_steps = 0
        self.after_scheduler = after_scheduler
        self.lr_history = []
        self._last_lr = 0.0 if self.n_warmup_steps > 0 else self.base_lr

        # Start from zero when warmup is enabled so the first steps truly ramp up.
        initial_lr = self._last_lr
        for pg in self._optimizer.param_groups:
            pg["lr"] = initial_lr

    def step(self, step=None, loss=None):
        """Step the scheduler once (typically per batch)."""
        self._update_learning_rate(step=step, loss=loss)

    def _update_learning_rate(self, step, loss=None):
        """Learning rate scheduling per step."""
        # Determine current step index (1-based as in your reference)
        if step is None:
            self.n_steps += 1
            step = self.n_steps
        else:
            step = int(step)
            self.n_steps = step

        # Warmup phase
        if self.n_warmup_steps > 0 and step <= self.n_warmup_steps:
            lr = self.base_lr * float(step) / float(max(self.n_warmup_steps, 1))
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = lr
            self._last_lr = lr
            self.lr_history.append(lr)
            return

        # After warmup
        # Delegate scheduling policy. Common schedulers accept an "epoch" argument;
        # here we pass the global step to mimic per-step scheduling.
        if self.after_scheduler is not None:
            scheduler_step = max(0, step - self.n_warmup_steps)
            try:
                self.after_scheduler.step(scheduler_step)
            except TypeError:
                # Some schedulers expect no arguments
                self.after_scheduler.step()

        # Read back current lr from optimizer
        lr = self._optimizer.param_groups[-1]["lr"]
        self._last_lr = float(lr)
        self.lr_history.append(self._last_lr)

    def get_last_lr(self):
        return [float(self._last_lr)]

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "n_warmup_steps": self.n_warmup_steps,
            "n_steps": self.n_steps,
            "lr_history": list(self.lr_history),
            "after_scheduler": (
                self.after_scheduler.state_dict()
                if self.after_scheduler is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict):
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.n_warmup_steps = int(state_dict.get("n_warmup_steps", self.n_warmup_steps))
        self.n_steps = int(state_dict.get("n_steps", self.n_steps))
        self.lr_history = list(state_dict.get("lr_history", self.lr_history))
        if (
            self.after_scheduler is not None
            and state_dict.get("after_scheduler") is not None
        ):
            self.after_scheduler.load_state_dict(state_dict["after_scheduler"])
        # Re-apply current lr for safety
        last_lr = float(self.lr_history[-1]) if self.lr_history else self.base_lr
        self._last_lr = last_lr
        for pg in self._optimizer.param_groups:
            pg["lr"] = last_lr


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return {}
        return json.loads(text)


def load_data_registry(config_data_files) -> dict:
    registry = {"datasets": {}}
    for p in config_data_files:
        cfg = _read_json(p)
        file_train_defaults = cfg.get("train_defaults", {}) or {}

        for name, spec in (cfg.get("datasets") or {}).items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Invalid dataset spec in '{p}' for key '{name}': expected a JSON object, got {type(spec).__name__}."
                )
            merged = dict(spec)
            if file_train_defaults:
                td = dict(file_train_defaults)
                td.update(merged.get("train_defaults", {}) or {})
                merged["train_defaults"] = td
            registry["datasets"][name] = merged

        if "default_loader" in cfg:
            registry["default_loader"] = cfg["default_loader"]
    return registry


def load_model_registry(config_model_files) -> dict:
    merged = {}
    for p in config_model_files:
        merged = _deep_merge_dict(merged, _read_json(p))
    return merged


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_run_command(exp_root: str) -> None:
    cmd = shlex.join([sys.executable, *sys.argv])
    cmd_path = os.path.join(exp_root, "command.sh")
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(cmd + "\n")


def _to_jsonable(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def args_to_json_dict(args) -> dict:
    return {
        key: _to_jsonable(value)
        for key, value in vars(args).items()
        if not str(key).startswith("_")
    }


def _infer_feature_dim(x):
    if torch.is_tensor(x) and x.ndim >= 1:
        return int(x.shape[-1])
    return None


def maybe_infer_feature_dims(args, train_dataloader):
    try:
        it = iter(train_dataloader)
        for _ in range(10):
            batch = next(it)
            if batch is None:
                continue
            X, _y = batch
            if not isinstance(X, (list, tuple)) or len(X) != len(args.modalities):
                return
            for i, mod in enumerate(args.modalities):
                dim = _infer_feature_dim(X[i])
                if dim is None:
                    continue
                if mod in ["text", "tabular"]:
                    args.t_feature_dim = dim
                elif mod == "vision":
                    args.v_feature_dim = dim
                elif mod in ["audio", "atimeseries"]:
                    args.a_feature_dim = dim
            return
    except StopIteration:
        return


def cli_arg_provided(args, key: str) -> bool:
    return key in getattr(args, "_cli_overrides", set())


def set_arg_unless_cli_overridden(args, key: str, value) -> None:
    if value is None or cli_arg_provided(args, key):
        return
    setattr(args, key, value)


def apply_train_defaults(args, train_defaults: dict) -> None:
    defaults = {
        "optimizer": "adam",
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 1e-4,
        "early_stop_patience": 8,
        "early_stop_min_delta": 0.0,
        "learning_rate": 1e-4,
        "epochs": 30,
    }
    defaults.update(train_defaults or {})
    for key, value in defaults.items():
        set_arg_unless_cli_overridden(args, key, value)


def auto_select_fusion(args, model_registry: dict):
    if args.fusion_method != "auto":
        return
    auto_map = model_registry.get("fusion_auto_by_num_modalities", {}) or {}
    picked = auto_map.get(str(len(args.modalities)))
    if picked is None:
        picked = "MT" if len(args.modalities) >= 2 else "Uni"
    args.fusion_method = picked


def apply_model_arg_overrides(args, model_registry: dict):
    fusion_overrides = model_registry.get("fusion_arg_overrides", {}) or {}
    spec = fusion_overrides.get(args.fusion_method) or {}
    resolved = {}
    resolved.update(spec.get("defaults", {}) or {})
    resolved.update(((spec.get("datasets") or {}).get(args.dataset)) or {})
    for key, value in resolved.items():
        set_arg_unless_cli_overridden(args, key, value)


def normalize_fusion_method(args):
    if args.fusion_method == "EFT":
        args.fusion_method = "TF"


def resolve_value(v, context: dict):
    if isinstance(v, str) and v.startswith("$"):
        key = v[1:]
        if key not in context:
            raise KeyError(f"Unknown placeholder '{v}' in model config.")
        return context[key]
    return v


def instantiate_from_spec(spec, context: dict, class_registry: dict):
    if spec is None:
        return None
    if not isinstance(spec, dict) or "class" not in spec:
        raise ValueError(f"Invalid module spec: {spec}")
    cls_name = spec["class"]
    if cls_name not in class_registry:
        raise KeyError(f"Unknown class '{cls_name}' in model config.")
    cls = class_registry[cls_name]
    kwargs = {
        k: resolve_value(v, context) for k, v in (spec.get("kwargs") or {}).items()
    }
    return cls(**kwargs)


class ConfigNode(dict):
    """dict with attribute access, preserving .get()."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value
