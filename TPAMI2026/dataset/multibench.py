from dataset.paths import load_catalog
import os
import json
import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset, DataLoader, default_collate

# Local imports
from dataset.affect.get_data import Affect, collate_fn_timeseries
from dataset.mimic.get_data import MIMIC

num_workers = 4


class MultiBench(Dataset):
    def __init__(self, dataset: str, split: str, **dataset_kwargs):
        super().__init__()
        catalog_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "catalog.json"
        )
        self.catalog = load_catalog()
        self.collate_fn = None
        if dataset == "mimic":
            self.dataset = MIMIC(
                self.catalog[dataset]["path"], split=split, **dataset_kwargs
            )
            self.n_classes = self.dataset.n_classes
        elif dataset in ["humor", "sarcasm", "mosi", "mosei"]:
            self.dataset = Affect(
                self.catalog[dataset]["path"],
                dataset=dataset,
                split=split,
                **dataset_kwargs,
            )
            self.collate_fn = self.collate_fn_affect
            self.n_classes = 2
        self.collate_fn = self.collate_fn_affect
        self.modalities = self.dataset.modalities

    @staticmethod
    def collate_fn_affect(inputs: List, max_seq_length: int = None):
        """Handles a list of labelled timeseries data with eventually different lengths.
        Args:
             inputs: list (X, y) where X is a list of modalities and y a label
             max_seq_length: if set, pads all timeseries to `max_seq_length`.
                Otherwise, all sequences are padded to the maximum sequence length in this batch.
        Output:
            (X, y) where X is a list of modalities with shape (*, T, p) and y a label with shape (*, 1).
            If `max_seq_length` is set then T == `max_seq_length`.
        """
        labels = np.array([y for (X, y) in inputs]).reshape(
            len(inputs),
        )
        X = collate_fn_timeseries([X for (X, y) in inputs], max_seq_length)
        return X, torch.tensor(labels)

    def __getitem__(self, i):
        X, y = self.dataset[i]
        return X, y

    def __len__(self):
        return len(self.dataset)


def get_collate(dataset):
    return getattr(dataset, "collate_fn", default_collate)


def multibench_get_loader(
    dataset_name: str, batch_size: int, modalities: List[str] = None, **dataset_kwargs
):
    train_dataset = MultiBench(
        dataset_name, split="train", modalities=modalities, **dataset_kwargs
    )
    test_dataset = MultiBench(
        dataset_name, split="test", modalities=modalities, **dataset_kwargs
    )
    valid_datset = MultiBench(
        dataset_name, split="val", modalities=modalities, **dataset_kwargs
    )
    n_classes = train_dataset.n_classes if hasattr(train_dataset, "n_classes") else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate(train_dataset),
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_collate(test_dataset),
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_datset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_collate(valid_datset),
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, valid_dataloader, n_classes
