import pandas as pd
from torch import Tensor
from pathlib import Path
from typing import Union, Tuple
from interfaces import IProcessor
from torch.utils.data import Dataset
from constants import path_column, class_column


class Data(Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            file_processor: IProcessor,
            text_processor: IProcessor,
            cls_mapper: dict
            ) -> None:
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.file_processor = file_processor
        self.text_processor = text_processor
        self.cls_mapper = cls_mapper

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        row = self.df.iloc[idx]
        file = row[path_column]
        text = row[class_column]
        cls = self.cls_mapper[text]
        text, mask = self.text_processor.process(text)
        speech = self.text_processor.process(file)
        return speech, text, mask, cls

    def __len__(self):
        return self.df.shape[0]
