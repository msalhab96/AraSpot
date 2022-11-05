import pandas as pd
from torch import Tensor
from pathlib import Path
from typing import Union, Tuple
from interfaces import IProcessor
from torch.utils.data import Dataset, DataLoader
from constants import path_column, class_column
from processors import get_augmenter, get_speech_processor


class Data(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            file_processor: IProcessor,
            cls_mapper: dict
            ) -> None:
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.file_processor = file_processor
        self.cls_mapper = cls_mapper

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        row = self.df.iloc[idx]
        file = row[path_column]
        text = row[class_column]
        cls = self.cls_mapper[text]
        speech = self.file_processor.process(file)
        return speech, cls

    def __len__(self):
        return self.df.shape[0]


def get_loaders(cfg, cls_mapper: dict):
    train_data = Data(
        data_path=cfg.train_path,
        file_processor=get_speech_processor(
            cfg, augment=True, augmenter=get_augmenter(cfg)
            ),
        cls_mapper=cls_mapper
    )
    test_data = Data(
        data_path=cfg.test_path,
        file_processor=get_speech_processor(cfg, augment=False),
        cls_mapper=cls_mapper
    )
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True
        )
    test_loader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=True
        )
    return train_loader, test_loader
