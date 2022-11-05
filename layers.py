import math
import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from functools import lru_cache


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


class MHSA(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert enc_dim % h == 0, 'enc_dim is not divisible by h'
        self.fc_key = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_query = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_value = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.proj_fc = nn.Linear(
            in_features=2 * enc_dim,
            out_features=enc_dim,
        )
        self.lnorm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.enc_dim = enc_dim
        self.h = h
        self.dk = enc_dim // h
        self.sqrt_dk = math.sqrt(self.dk)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
        Returns:
            Tensor: The scaled attention weights of shape
            [B * h, Tq, Tk]
        """
        result = torch.matmul(Q, K)
        result = result / self.sqrt_dk
        return self.softmax(result)

    def perform_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor
            ) -> Tensor:
        """Performs multi-head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
            V (Tensor): The Value tensor of shape [h * B, Tk, dk]
        Returns:
            Tuple[Tensor, Tensor]: The attention matrix of shape
            [B * h, Tq, Tk] and the scaled attention value of
            shape [B * h, Tq, dk].
        """
        att = self._get_scaled_att(Q, K)
        result = torch.matmul(att, V)
        return att, result

    @lru_cache(maxsize=2)
    def get_positionals(self, max_length: int) -> Tensor:
        """Create Positionals tensor to be added to the input
        Args:
            max_length (int): The maximum length of the positionals sequence.
        Returns:
            Tensor: Positional tensor
        """
        result = torch.zeros(max_length, self.enc_dim, dtype=torch.float)
        for pos in range(max_length):
            for i in range(0, self.enc_dim, 2):
                denominator = pow(10000, 2 * i / self.enc_dim)
                result[pos, i] = math.sin(pos / denominator)
                result[pos, i + 1] = math.cos(pos / denominator)
        return result

    def _reshape(self, *args) -> List[Tensor]:
        """Reshabes all the given list of tensor
        from [B, T, N] to [B, T, h, dk]
        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.contiguous().view(-1, item.shape[1], self.h, self.dk)
            for item in args
        ]

    def _pre_permute(self, *args) -> List[Tensor]:
        """Permutes all the given list of tensors
        from [B, T, h, dk] to become [h, B, T, dk].
        Returns:
            List[Tensor]: List of all permuted tensors.
        """
        return [
            item.permute(2, 0, 1, 3)
            for item in args
        ]

    def _change_dim(self, *args) -> List[Tensor]:
        """Changes the dimensionality of all passed tensores
        from [B, T, N] to [B * h, T, dk]
        Returns:from functools import lru_cache
            List[Tensor]: List of the modified tensors.
        """
        result = self._reshape(*args)  # [B, T, h, dk]
        result = self._pre_permute(*result)  # [h, B, T, dk]
        return [
            item.permute(1, 0, 2, 3).contiguous().view(
                -1, item.shape[2], item.shape[3]
                )
            for item in result
        ]

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the input into multi-head attention
        Args:
            inp (Tensor): The input tensor
        Returns:
            Tensor: The result after adding it to positionals
            and passing it through multi-head self-attention
        """
        pos = self.get_positionals(inp.shape[1])
        pos = pos.unsqueeze(dim=0).to(inp.device)
        inp = pos + inp
        out = self.lnorm(inp)
        [b, s, _] = inp.shape
        K = self.fc_key(inp)
        Q = self.fc_query(inp)
        V = self.fc_value(inp)
        (Q, K, V) = self._change_dim(Q, K, V)  # [h * B, T, dk]
        K = K.permute(0, 2, 1)  # [h, T, B, dk]
        _, result = self.perform_att(Q, K, V)
        result = result.view(b, self.h, s, self.dk)
        result = result.permute(0, 2, 1, 3)
        result = result.contiguous().view(b, s, -1)
        result = torch.cat([inp, result], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)
        return inp + out


class ConformerBlock(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.ff1 = FeedForwardModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            p_dropout=p_dropout,
            residual_scaler=residual_scaler
        )
        self.mhsa = MHSA(
            enc_dim=enc_dim,
            h=h, p_dropout=p_dropout, device=device
        )
        self.conv = ConvModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            kernel_size=kernel_size,
            p_dropout=p_dropout
            )
        self.ff2 = FeedForwardModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            p_dropout=p_dropout,
            residual_scaler=residual_scaler
        )
        self.lnorm = nn.LayerNorm(enc_dim)

    def forward(self, inp: Tensor):
        out = self.ff1(inp)
        out = self.mhsa(out)
        out = self.conv(out)
        out = self.ff2(out)
        out = self.lnorm(out)
        return out


class Conformer(nn.Module):
    def __init__(
            self,
            n_layers: int,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                enc_dim=enc_dim,
                kernel_size=kernel_size,
                scaling_factor=scaling_factor,
                residual_scaler=residual_scaler,
                device=device,
                p_dropout=p_dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
