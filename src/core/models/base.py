"""
Classe base abstrata para modelos de classificação.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Classe base abstrata para modelos de classificação.

    Define a interface comum que todos os modelos devem implementar:
    - forward(): retorna logits de classificação
    - get_representation(): retorna representação latente (para visualização)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.

        Parâmetros
        ----------
        x : torch.Tensor
            Input do modelo (batch_size, input_dim).

        Retorna
        -------
        torch.Tensor
            Logits de classificação (batch_size,) para classificação binária.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai a representação latente do modelo (ex: penúltima camada).

        Usado para visualização (t-SNE, UMAP, etc).

        Parâmetros
        ----------
        x : torch.Tensor
            Input do modelo (batch_size, input_dim).

        Retorna
        -------
        torch.Tensor
            Representação latente (batch_size, repr_dim).
        """
        pass
