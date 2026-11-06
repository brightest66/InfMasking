"""Portable locations for data and pretrained model files."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_catalog():
    data_root = (
        Path(os.environ.get("TPAMI_DATA_ROOT", PROJECT_ROOT / "data"))
        .expanduser()
        .resolve()
    )
    bert = os.environ.get(
        "TPAMI_BERT_MODEL", str(PROJECT_ROOT / "pretrained/bert-base-uncased")
    )
    catalog = json.loads((PROJECT_ROOT / "dataset/catalog.json").read_text())
    for spec in catalog.values():
        for key in ("path", "metadata"):
            if key in spec:
                spec[key] = str(data_root / spec[key])
        if "bert_model_path" in spec:
            spec["bert_model_path"] = bert
    return catalog
