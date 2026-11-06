#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for dataset in mosi mosei mimic mami mute; do
    "${PYTHON:-python}" "$project_root/scripts/run_experiment.py" --dataset "$dataset" "$@"
done
