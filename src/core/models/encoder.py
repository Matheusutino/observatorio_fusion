"""
Modelo Encoder (anteriormente MLP) para classificação binária.
"""
import torch
import torch.nn as nn

from .base import BaseModel


class Encoder(BaseModel):
    """
    Encoder feedforward para classificação binária.

    Arquitetura: Input → [Linear + LayerNorm + ReLU + Dropout]* → Linear(1)

    Loss: BCEWithLogitsLoss (apenas classificação)
    """

    HIDDEN_DIMS = [256, 128]

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        """
        Parâmetros
        ----------
        input_dim : int
            Dimensão do input (embedding fusionado).
        hidden_dims : list[int]
            Lista com dimensões das camadas ocultas.
        dropout : float
            Taxa de dropout.
        """
        super().__init__()
        _ = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.HIDDEN_DIMS[0]),
            nn.LayerNorm(self.HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_DIMS[0], self.HIDDEN_DIMS[1]),
            nn.LayerNorm(self.HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_DIMS[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        torch.Tensor
            Logits de classificação (batch_size,).
        """
        return self.net(x).squeeze(1)

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna a ativação da penúltima camada (antes do logit final).

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        torch.Tensor
            Representação latente (batch_size, hidden_dims[-1]).
        """
        self.eval()
        # net[-1] é o Linear final — passa por todas as camadas anteriores
        out = x
        for layer in list(self.net.children())[:-1]:
            out = layer(out)
        return out
