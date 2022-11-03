from pathlib import Path
from typing import Tuple, Union
from config import get_feat_args
from interfaces import IProcessor
from torch import Tensor
import torch
import torchaudio
from torchaudio import transforms
from constants import pad, eos, sos
from utils import load_json


class TextProcessor(IProcessor):
    def __init__(
            self,
            chars_maper: dict,
            max_len: int
            ) -> None:
        super().__init__()
        self.chars_maper = chars_maper
        self.max_len = max_len

    def process(self, text: str) -> Tuple[Tensor, Tensor]:
        tokens = [self.chars_maper[sos]]
        tokens += [self.chars_maper[c] for c in text]
        tokens += [self.chars_maper[eos]]
        length = len(tokens)
        diff = self.max_len - length
        tokens += [self.chars_maper[pad]] * diff
        mask = [False] * length + [True] * diff
        return torch.LongTensor(tokens), torch.BoolTensor(mask)


class FileProcessor(IProcessor):
    def __init__(
            self,
            sampling_rate: int,
            feature: str,
            feature_args: dict
            ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        if feature == 'mfcc':
            self.feat_extractor = transforms.MFCC(**feature_args)
        if feature == 'melspec':
            self.feat_extractor = transforms.MelSpectrogram(**feature_args)

    def _resample(self, x: Tensor, sr: int) -> Tensor:
        return transforms.Resample(
            orig_freq=sr, new_freq=self.sampling_rate
            )(x)

    def load(self, file_path: Union[str, Path]):
        x, sr = torchaudio.load(file_path)
        x = self._resample(x, sr)
        return x

    def process(self, file: Union[str, Path]):
        # TODO: add SpecAug
        x = self.load(file)
        x = self.feat_extractor(x)
        x = x.permute(0, 2, 1)
        return x.squeeze()


def get_text_processor(cfg):
    return TextProcessor(
        chars_mapper=load_json(cfg.chars_mapper),
        max_len=cfg.max_len
    )


def get_speech_processor(cfg):
    return FileProcessor(
        sampling_rate=cfg.sampling_rate,
        feature=cfg.feature,
        feature_args=get_feat_args(cfg)
    )


def get_processors(cfg):
    return (
        get_text_processor(cfg),
        get_speech_processor(cfg)
        )
