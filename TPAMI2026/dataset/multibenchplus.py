"""MAMI/MUTE loaders adapted from MultiBench++; original split order is preserved."""

import os
import pandas as pd
from openpyxl import load_workbook
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer
from dataset.paths import load_catalog

num_workers = 8


class MUTEDataset(Dataset):
    """
    Custom Dataset for the MUTE dataset.
    Handles loading images, text, and labels from an XLSX file.
    """

    def __init__(
        self, xlsx_file, root_dir, transform=None, modalities=["text", "vision"]
    ):
        """
        Args:
            xlsx_file (string): Path to the xlsx file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.modalities = modalities

        self.label_map = {"not-hate": 0, "hate": 1}
        self.annotations = self._load_annotations(xlsx_file)

    def _load_annotations(self, xlsx_file):
        # Read the workbook without pandas because some environments fail
        # inside DataFrame construction for this dataset's xlsx files.
        workbook = load_workbook(xlsx_file, read_only=True, data_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        rows = sheet.iter_rows(values_only=True)

        try:
            next(rows)
        except StopIteration:
            return []

        annotations = []
        for row in rows:
            if row is None or len(row) < 3:
                continue

            img_name, caption, raw_label = row[0], row[1], row[2]
            if raw_label not in self.label_map:
                continue

            annotations.append((img_name, caption, self.label_map[raw_label]))

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, caption, label = self.annotations[idx]

        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except (FileNotFoundError, IOError):
            raise FileNotFoundError(f"Missing or unreadable image: {img_path}")

        if not isinstance(caption, str):
            caption = ""

        label = torch.tensor(label, dtype=torch.long)
        X = []
        for mod in self.modalities:
            if mod == "text":
                X.append(caption)
            elif mod == "vision":
                X.append(image)
            else:
                raise ValueError(f"Unsupported modality: {mod}")
        return X, label


def mute_get_loader(dataset_name="mute", batch_size=8, modalities=["text", "vision"]):
    """
    Creates and returns the data loaders for the MUTE dataset.
    This version tokenizes text in batches and returns a list.
    """
    catalog_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "catalog.json"
    )
    catalog = load_catalog()
    tokenizer = BertTokenizer.from_pretrained(catalog[dataset_name]["bert_model_path"])
    base_dir = catalog[dataset_name]["metadata"]
    root_dir = os.path.join(base_dir, "Memes")
    train_file = os.path.join(base_dir, "train_hate.xlsx")
    val_file = os.path.join(base_dir, "valid_hate.xlsx")
    test_file = os.path.join(base_dir, "test_hate.xlsx")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = MUTEDataset(
        xlsx_file=train_file,
        root_dir=root_dir,
        transform=transform,
        modalities=modalities,
    )
    val_dataset = MUTEDataset(
        xlsx_file=val_file,
        root_dir=root_dir,
        transform=transform,
        modalities=modalities,
    )
    test_dataset = MUTEDataset(
        xlsx_file=test_file,
        root_dir=root_dir,
        transform=transform,
        modalities=modalities,
    )

    n_classes = len(train_dataset.label_map)

    def custom_collate_fn(batch):
        Xs, labels = zip(*batch)
        labels = torch.stack(labels, 0)
        mod_to_idx = {m: i for i, m in enumerate(modalities)}
        images = None
        encoded_text = None
        images = torch.stack([x_list[mod_to_idx["vision"]] for x_list in Xs], 0)
        texts = [x_list[mod_to_idx["text"]] for x_list in Xs]
        encoded_text = tokenizer.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        X = []
        for mod in modalities:
            if mod == "vision":
                X.append(images)
            elif mod == "text":
                X.append(encoded_text)
            else:
                raise ValueError(f"Unsupported modality in collate_fn: {mod}")
        return X, labels

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )

    return train_loader, test_loader, val_loader, n_classes


class MAMIDataset(Dataset):

    def __init__(
        self, annotations_file, img_dir, transform=None, modalities=["text", "vision"]
    ):

        self.data_frame = pd.read_csv(annotations_file, sep="\t")
        self.img_dir = img_dir
        self.transform = transform
        self.modalities = modalities

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.loc[idx, "file_name"]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing image: {img_path}")

        text = self.data_frame.loc[idx, "text"]

        if not isinstance(text, str):
            text = ""

        label = torch.tensor(self.data_frame.loc[idx, "label"], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        X = []
        for mod in self.modalities:
            if mod == "text":
                X.append(text)
            elif mod == "vision":
                X.append(image)
            else:
                raise ValueError(f"Unsupported modality: {mod}")

        return X, label


def mami_get_loader(dataset_name="mami", batch_size=8, modalities=["text", "vision"]):
    catalog_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "catalog.json"
    )
    catalog = load_catalog()
    tokenizer = BertTokenizer.from_pretrained(catalog[dataset_name]["bert_model_path"])
    base_dir = catalog[dataset_name]["metadata"]

    image_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def custom_collate_fn(batch):
        Xs, labels = zip(*batch)
        labels = torch.stack(labels, 0)

        mod_to_idx = {m: i for i, m in enumerate(modalities)}

        images = None
        encoded_text = None
        images = torch.stack([x_list[mod_to_idx["vision"]] for x_list in Xs], 0)
        texts = [x_list[mod_to_idx["text"]] for x_list in Xs]
        encoded_text = tokenizer.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        X = []
        for mod in modalities:
            if mod == "vision":
                X.append(images)
            elif mod == "text":
                X.append(encoded_text)
            else:
                raise ValueError(f"Unsupported modality in collate_fn: {mod}")
        return X, labels

    dataloaders = {}
    splits = ["train", "validation", "test"]

    root_dir = base_dir
    for split in splits:
        annotations_path = os.path.join(root_dir, f"{split}.tsv")

        if not os.path.exists(annotations_path):
            raise FileNotFoundError(annotations_path)

        img_dir_name = "test_images" if split == "test" else "training_images"
        img_dir = os.path.join(root_dir, img_dir_name)

        dataset = MAMIDataset(
            annotations_file=annotations_path,
            img_dir=img_dir,
            transform=image_transforms,
            modalities=modalities,
        )

        shuffle = split == "train"

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        dataloaders[split] = dataloader

    return dataloaders["train"], dataloaders["test"], dataloaders["validation"], 2
