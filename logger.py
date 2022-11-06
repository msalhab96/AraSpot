from pathlib import Path
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from interfaces import ILogger


class Logger(SummaryWriter, ILogger):
    def __init__(
            self, log_dir: Union[Path, str], *args, **kwargs
            ):
        super().__init__(log_dir, *args, **kwargs)
        self._records = {}

    def log(self):
        step = len(list(self._records.values())[0])
        msg = f'epoch {step}: '
        for key, value in self._records.items():
            self.add_scalar(key, value[-1], global_step=step)
            msg += f'| {key}: {value[-1]}'
        print(msg)
        print('=' * len(msg))

    def record(self, key: str, value: Union[float, int]):
        if key in self._records:
            self._records[key].append(value)
        else:
            self._records[key] = [value]


def get_logger(cfg):
    return Logger(cfg.logdir)
