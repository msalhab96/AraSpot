from torch import nn
from torch import Tensor


class GRUBlock(nn.Module):
    def __init__(
            self,
            in_size: int,
            hidden_size: int,
            bidirectional: bool
            ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.lnorm = nn.LayerNorm(
            normalized_shape=2*hidden_size if bidirectional else hidden_size
            )

    def forward(self, x: Tensor):
        out, hn = self.gru(x)
        out = self.lnorm(out)
        return out, hn


class GRUStack(nn.Module):
    def __init__(
            self,
            n_layers: int,
            in_size: int,
            hidden_size: int,
            bidirectional: bool
            ):
        super().__init__()
        self.layers = nn.ModuleList(
            GRUBlock(
                in_size=in_size if i == 0 else hidden_size * (
                    2 if bidirectional else 1
                    ),
                hidden_size=hidden_size,
                bidirectional=bidirectional
            )
            for i in range(n_layers)
        )

    def forward(self, x: Tensor):
        out = x
        for layer in self.layers:
            out, hn = layer(out)
        hn = hn.permute(1, 0, 2)
        hn = hn.contiguous().view(hn.shape[0], 1, -1)
        return out, hn
