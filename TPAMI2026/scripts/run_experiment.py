"""Train and evaluate InfMasking with a dataset-specific configuration."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command_for(dataset, seed, device, output_root, stop_after_epochs=0):
    config_path = ROOT / "configs/experiments" / f"{dataset}.json"
    if not config_path.exists():
        raise ValueError(f"No configuration for {dataset}")
    cfg = json.loads(config_path.read_text())
    cmd = [sys.executable, str(ROOT / "main.py")]
    for key, value in cfg.items():
        if key == "use_masking":
            if value:
                cmd.append("--use_masking")
        elif isinstance(value, list):
            cmd += ["--" + key] + list(map(str, value))
        else:
            cmd += [
                "--" + key,
                str(value).lower() if isinstance(value, bool) else str(value),
            ]
    cmd += [
        "--random_seed",
        str(seed),
        "--device",
        device,
        "--exp_root",
        str(Path(output_root).resolve()),
        "--exp_name",
        "TPAMI2026-infmask",
        "--stop_after_epochs",
        str(stop_after_epochs),
        "--tb_disable",
    ]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=["mosi", "mosei", "mimic", "mami", "mute"], required=True
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 11, 111])
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--output-root",
        default=os.environ.get("TPAMI_OUTPUT_ROOT", str(ROOT / "outputs")),
    )
    p.add_argument("--stop-after-epochs", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for seed in args.seeds:
        cmd = command_for(
            args.dataset,
            seed,
            args.device,
            args.output_root,
            stop_after_epochs=args.stop_after_epochs,
        )
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
