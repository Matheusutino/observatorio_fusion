"""
Modelo Autoencoder com classificação para classificação binária.
"""
import torch
import torch.nn as nn

from .base import BaseModel


class Autoencoder(BaseModel):
    """
    Autoencoder com branch de classificação.

    Arquitetura:
    - Encoder: Input → [Linear + LayerNorm + ReLU + Dropout]* → Latent
    - Decoder: Latent → [Linear + LayerNorm + ReLU + Dropout]* → Reconstruction
    - Classifier: Latent → Linear(1)

    Losses:
    - Reconstruction: MSELoss
    - Classification: BCEWithLogitsLoss
    """

    ENCODER_DIMS = [256, 128]
    LATENT_DIM = 64

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float,
    ):
        """
        Parâmetros
        ----------
        input_dim : int
            Dimensão do input (embedding fusionado).
        hidden_dims : list[int]
            Lista com dimensões das camadas ocultas do encoder.
        latent_dim : int
            Dimensão do espaço latente.
        dropout : float
            Taxa de dropout.
        """
        super().__init__()
        _ = (hidden_dims, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.to_latent = nn.Linear(self.ENCODER_DIMS[1], self.LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[1], self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], input_dim),
        )

        # ========== Classifier ==========
        self.classifier = nn.Linear(self.LATENT_DIM, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder: input → latent representation."""
        h = self.encoder(x)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: latent → reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor]
            (logits, reconstruction)
            - logits: (batch_size,) para classificação
            - reconstruction: (batch_size, input_dim)
        """
        z = self.encode(x)
        logits = self.classifier(z).squeeze(1)
        reconstruction = self.decode(z)
        return logits, reconstruction

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna a representação latente (espaço latente do encoder).

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        torch.Tensor
            Representação latente (batch_size, latent_dim).
        """
        self.eval()
        return self.encode(x)
