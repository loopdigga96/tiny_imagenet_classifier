import os
from typing import Callable, Optional
from PIL import Image
from pathlib import Path

from torch.utils.data.dataset import Dataset
from pandas import DataFrame
import pandas as pd

Transform = Callable[[Image.Image], Image.Image]


class TinyImagenetDataset(Dataset):
    _transform: Optional[Transform]
    _root: Path
    _df: DataFrame

    def __init__(self, path, folders_to_num, val_labels, transform: Optional[Transform] = None):
        self._transform = transform
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory.")
        all_files = [
            os.path.join(r, fyle)
            for r, d, f in os.walk(path)
            for fyle in f
            if ".JPEG" in fyle
        ]
        labels = [
            folders_to_num.get(
                os.path.basename(f).split("_")[0],
                folders_to_num.get(val_labels.get(os.path.basename(f))),
            )
            for f in all_files
        ]
        self._df = pd.DataFrame({"path": all_files, "label": labels})

    def __getitem__(self, index: int) -> dict:
        path, label = self._df.loc[index, :]
        image = Image.open(path).convert("RGB")
        if self._transform:
            image = self._transform(image)
        return {'image': image, 'label': label}

    def __len__(self) -> int:
        return len(self._df)


class TinyImagenetTestSet(Dataset):
    def __init__(self, path, transform: Optional[Transform] = None):
        self._transform = transform
        self.path = path
        self.test_files = os.listdir(path)

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        filename = self.test_files[index]
        full_path = os.path.join(self.path, filename)
        image = Image.open(full_path).convert("RGB")
        image = self._transform(image)

        return {'image': image, 'filename': filename}
