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
