import torch.nn as nn
from torch import Tensor


class FeedForwardModule(nn.Module):
    """Implements the feed forward module in the conformer block
    where the module consists of the below
    1. Layer Norm
    2. Linear Layer
    3. Swish Activation
    4. Dropout
    5. Linear Layer
    6. Dropout
    Args:
        enc_dim (int): The encoder dimensionality
        scaling_factor (int): The scaling factor of the linear layer
        p_dropout (float): The dropout probability
        residual_scaler (float, optional): The residual scaling.
        Defaults to 0.5.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            p_dropout: float,
            residual_scaler=0.5
            ) -> None:
        super().__init__()
        self.residual_scaler = residual_scaler
        scaled_dim = scaling_factor * enc_dim
        self.lnorm = nn.LayerNorm(enc_dim)
        self.fc1 = nn.Linear(
            in_features=enc_dim,
            out_features=scaled_dim
        )
        self.fc2 = nn.Linear(
            in_features=scaled_dim,
            out_features=enc_dim
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the given inp through the feed forward
        module
        Args:
            inp (Tensor): the input to the feed forward module
            with shape [B, M, N] where B is the batch size, M
            is the maximum length, and N is the encoder dim
        Returns:
            Tensor: The result of the forward module
        """
        out = self.lnorm(inp)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.residual_scaler * inp + out


class ConvModule(nn.Module):
    """Implements the convolution module
    where it contains the following layers
    1. Layernorm
    2. Pointwise Conv
    3. Gate Linear unit
    4. 1D Depthwise conv
    5. BatchNorm
    6. Swish Activation
    7. Pointwise Conv
    8. Dropout
    Args:
        enc_dim (int): The encoder dimensionality.
        scaling_factor (int): The scaling factor of the conv layer.
        kernel_size (int): The convolution kernel size.
        p_dropout (float): The dropout probability.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            kernel_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(enc_dim)
        n_scaled_channels = enc_dim * scaling_factor
        assert (kernel_size - 1) % 2 == 0, 'kernel_size - 1 \
            must be divisable by 2 -odd'
        padding_size = (kernel_size - 1) // 2
        self.pwise_conv1 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=n_scaled_channels,
            kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=enc_dim
        )
        self.bnorm = nn.BatchNorm1d(enc_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)
        self.pwise_conv2 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=1
        )

    def forward(self, inp: Tensor) -> Tensor:
        out = self.lnorm(inp)
        out = out.permute(0, 2, 1)
        out = self.pwise_conv1(out)
        out = self.glu(out)
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.swish(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        return out + inp
