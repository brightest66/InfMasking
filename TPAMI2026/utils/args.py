"""Arguments for the five supervised InfMasking experiments."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["mosi", "mosei", "mimic", "mami", "mute"], default="mosi"
    )
    parser.add_argument(
        "--fusion_method", choices=["auto", "MT", "MT2"], default="auto"
    )
    parser.add_argument("--modalities", nargs="+")
    parser.add_argument(
        "--bert_model",
        default=os.environ.get(
            "TPAMI_BERT_MODEL", str(ROOT / "pretrained/bert-base-uncased")
        ),
    )
    for name, value in [
        ("edim", 768),
        ("multibench_dim", 40),
        ("num_heads", 1),
        ("n_classes", -1),
        ("a_feature_dim", 768),
        ("v_feature_dim", 768),
        ("t_feature_dim", 768),
        ("task", 7),
        ("batch_size", 64),
        ("epochs", 30),
        ("random_seed", 1),
        ("warmup_steps", -1),
        ("early_stop_min_epochs", 0),
        ("cosine_t0_epochs", 4),
        ("cosine_t_mult", 2),
        ("num_mask", 5),
        ("MT2_num_layers", 2),
        ("stop_after_epochs", 0),
        ("tb_log_interval", 50),
    ]:
        parser.add_argument("--" + name, type=int, default=value)
    for name, value in [
        ("learning_rate", 1e-3),
        ("weight_decay", 1e-4),
        ("mask_ratio", 0.5),
        ("mask_lambda", 1.0),
        ("MT_attn_dropout", 0.2),
        ("MT2_mlp_ratio", 4.0),
        ("MT2_dropout", 0.2),
    ]:
        parser.add_argument("--" + name, type=float, default=value)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_single", type=str2bool, default=True)
    parser.add_argument("--use_masking", action="store_true")
    parser.add_argument("--mask_objective", choices=["infmask"], default="infmask")
    parser.add_argument(
        "--loss_weights", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0]
    )
    parser.add_argument("--tb_disable", action="store_true")
    parser.add_argument("--exp_root", default=str(ROOT / "outputs"))
    parser.add_argument("--exp_name", default="")
    args = parser.parse_args()
    args._cli_overrides = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    # Compatibility fields consumed by shared experiment and loader helpers.
    args.fold = 0
    args.fold_use = False
    args.clean_validation = False
    if not args.use_masking or not args.use_single:
        parser.error(
            "InfMasking requires --use_masking and --use_single true; use scripts/run_experiment.py"
        )
    if args.stop_after_epochs < 0:
        parser.error("--stop_after_epochs must be nonnegative")
    return args
