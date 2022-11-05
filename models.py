import os
from config import get_model_args
from layers import Conformer
from torch import Tensor
from torch import nn
import torch


class ConformerGRU(nn.Module):
    def __init__(
            self,
            feat_size: int,
            n_layers: int,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            bidirectional: bool,
            n_classes: int,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.conf = Conformer(
            n_layers=n_layers,
            enc_dim=enc_dim,
            h=h,
            kernel_size=kernel_size,
            scaling_factor=scaling_factor,
            residual_scaler=residual_scaler,
            device=device,
            p_dropout=p_dropout
        )
        self.gru = nn.GRU(
            input_size=enc_dim,
            hidden_size=enc_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.pred_fc = nn.Linear(
            in_features=2 * enc_dim if bidirectional else enc_dim,
            out_features=n_classes
        )
        self.fc = nn.Linear(
            in_features=feat_size, out_features=enc_dim
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.conf(out)
        out, h = self.gru(out)
        h = h.permute(1, 0, 2).view(x.shape[0], -1)
        return self.pred_fc(h)


def get_model(cfg, n_classes):
    model = ConformerGRU(
        **get_model_args(cfg, n_classes)
    )
    if os.path.exists(cfg.ckpt_path) is True:
        model.load_state_dict(torch.load(cfg.ckpt_path))
        print(f'{cfg.ckpt_path} loadded!')
    return model
