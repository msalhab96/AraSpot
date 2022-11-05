import torch.nn as nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lsm = nn.LogSoftmax(dim=-1)
        self.nlll = nn.NLLLoss()

    def forward(self, preds: Tensor, target: Tensor):
        preds = self.lsm(preds)
        return self.nlll(preds, target)


def get_criterion(cfg, pad_idx):
    return Loss(cfg.alpha, pad_idx)
