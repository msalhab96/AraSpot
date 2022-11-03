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


class Encoder(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_conv: int,
            kernel_size: int,
            n_channels: int,
            n_layers: int,
            hidden_size: int,
            bidirectional: bool,
            enc_p_dropout: float
            ):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(
                in_channels=1 if i == 0 else n_channels * i,
                out_channels=1 if i == n_conv - 1 else n_channels * (1 + i),
                kernel_size=kernel_size
            )
            for i in range(n_conv)
        )
        in_size = n_mels - n_conv * (kernel_size - 1)
        self.grus = GRUStack(
            n_layers=n_layers,
            in_size=in_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(enc_p_dropout)

    def forward(self, x: Tensor):
        out = x.unsqueeze(dim=1)
        for layer in self.conv_layers:
            out = layer(out)
        out = out.squeeze(dim=1)
        out = self.dropout(out)
        out, hn = self.grus(out)
        return out, hn
