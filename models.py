from layers import Decoder, Encoder
from torch import Tensor
from torch import nn


class ConvSeq2Seq(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_conv: int,
            kernel_size: int,
            n_channels: int,
            voc_size: int,
            emb_dim: int,
            n_classes: int,
            n_layers: int,
            hidden_size: int,
            bidirectional: bool,
            enc_hidden_size: int,
            dec_hidden_size: int,
            dec_p_dropout: float,
            enc_p_dropout: float
            ):
        super().__init__()
        self.enc = Encoder(
            n_mels=n_mels,
            n_conv=n_conv,
            kernel_size=kernel_size,
            n_channels=n_channels,
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            enc_p_dropout=enc_p_dropout
        )
        dsize = enc_hidden_size * 2
        self.dec = Decoder(
            vocab_size=voc_size,
            emb_dim=emb_dim,
            enc_hidden_size=dsize if bidirectional else enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            p_dropout=dec_p_dropout
        )
        self.pred_fc = nn.Linear(
            in_features=dsize if bidirectional else enc_hidden_size,
            out_features=n_classes
        )

    def forward(self, x: Tensor, y: Tensor):
        out, hn = self.enc(x)
        result = self.dec(y, hn)
        hn = hn.squeeze(dim=1)
        preds = self.pred_fc(hn)
        return out, preds, result
