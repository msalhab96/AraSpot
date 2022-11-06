import os
from pathlib import Path
import random
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
            chars_mapper: dict,
            max_len: int
            ) -> None:
        super().__init__()
        self.chars_maper = chars_mapper
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


class Augmentor(IProcessor):
    def __init__(
            self,
            max_freq_len: int,
            max_time_len: int,
            n_mask: int,
            p_aug: float,
            sample_rate: int,
            noise_path
            ) -> None:
        super().__init__()
        self.fade_shapes = [
            'quarter_sine',
            'half_sine',
            'linear',
            'logarithmic',
            'exponential'
            ]
        self.spec_aug = transforms.FrequencyMasking(
            freq_mask_param=max_freq_len
        )
        self.time_aug = transforms.TimeMasking(
            time_mask_param=max_time_len
        )
        self.n_mask = n_mask
        self.p_aug = p_aug
        self._noise = []
        for file in os.listdir(noise_path):
            x, sr = torchaudio.load(os.path.join(noise_path, file))
            x = transforms.Resample(sr, sample_rate)(x)
            self._noise.append(x)
        self._noise = torch.hstack(self._noise)
        self._max_noise_idx = self._noise.shape[1]

    def rand_vol_gain(self, x: Tensor):
        gain = 2 * max(0.1, random.random())
        return transforms.Vol(gain=gain)(x)

    def rand_fade(self, x: Tensor):
        max_len = x.shape[1]
        shape = random.choice(self.fade_shapes)
        fade_in_len = random.randint(0, max_len)
        fade_out_len = random.randint(0, max_len)
        return transforms.Fade(
            fade_in_len=fade_in_len,
            fade_out_len=fade_out_len,
            fade_shape=shape
            )(x)

    def add_bg_noise(self, x):
        gain = random.random()
        length = x.shape[1]
        start = random.randint(0, self._max_noise_idx)
        end = random.randint(start, min(start + length, self._max_noise_idx))
        segment_len = end - start
        start_freedom = length - segment_len
        start_idx = random.randint(0, start_freedom)
        noise = gain * self._noise[:, start: end]
        x[:, start_idx: start_idx + segment_len] += noise
        return x

    def spec_mask(self, x: Tensor):
        for _ in range(self.n_mask):
            x = self.spec_aug(x)
        return x

    def time_mask(self, x: Tensor):
        for _ in range(self.n_mask):
            x = self.time_aug(x)
        return x

    def _apply(self, x, func):
        if random.random() > self.p_aug:
            x = func(x)
        return x

    def _time_aug(self, x: Tensor):
        ops = [
            self.add_bg_noise,
            self.rand_fade,
            self.rand_vol_gain
            ]
        random.shuffle(ops)
        for op in ops:
            x = self._apply(x, op)
        return x

    def _spec_aug(self, x: Tensor):
        x = self._apply(x, self.spec_mask)
        x = self._apply(x, self.time_mask)
        return x

    def process(self, x: Tensor, time=False, spec=False) -> Tensor:
        if time is True:
            return self._time_aug(x)
        if spec is True:
            return self._spec_aug(x)
        return x


class FileProcessor(IProcessor):
    def __init__(
            self,
            sampling_rate: int,
            feature: str,
            feature_args: dict,
            augmenter=None,
            augment=False
            ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        if feature == 'mfcc':
            self.feat_extractor = transforms.MFCC(**feature_args)
        elif feature == 'melspec':
            self.feat_extractor = transforms.MelSpectrogram(**feature_args)
        else:
            raise AttributeError
        self.augment = augment
        self.augmenter = augmenter

    def _resample(self, x: Tensor, sr: int) -> Tensor:
        return transforms.Resample(
            orig_freq=sr, new_freq=self.sampling_rate
            )(x)

    def load(self, file_path: Union[str, Path]):
        x, sr = torchaudio.load(file_path)
        x = self._resample(x, sr)
        return x

    def process(self, file: Union[str, Path]):
        x = self.load(file)
        if self.augment:
            x = self.augmenter.process(x, time=True)
        x = self.feat_extractor(x)
        x = x.permute(0, 2, 1)
        if self.augment:
            x = self.augmenter.process(x, spec=True)
        return x.squeeze()


def get_augmenter(cfg):
    return Augmentor(
        max_freq_len=cfg.max_freq_len,
        max_time_len=cfg.max_time_len,
        n_mask=cfg.n_mask,
        p_aug=cfg.p_aug,
        sample_rate=cfg.sample_rate,
        noise_path=cfg.noise_path
    )


def get_text_processor(cfg):
    return TextProcessor(
        chars_mapper=load_json(cfg.chars_mapper),
        max_len=cfg.max_len
    )


def get_speech_processor(cfg, augment=False, augmenter=None):
    return FileProcessor(
        sampling_rate=cfg.sample_rate,
        feature=cfg.feature,
        feature_args=get_feat_args(cfg),
        augmenter=augmenter,
        augment=augment
    )


def get_processors(cfg):
    return (
        get_text_processor(cfg),
        get_speech_processor(cfg)
        )
