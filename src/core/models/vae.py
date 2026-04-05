"""
Modelo Variational Autoencoder (VAE) com classificação.
"""
import torch
import torch.nn as nn

from .base import BaseModel


class VAE(BaseModel):
    """
    Variational Autoencoder com branch de classificação.

    Arquitetura:
    - Encoder: Input → [Linear + LayerNorm + ReLU + Dropout]* → (mu, logvar)
    - Reparameterization: z = mu + eps * exp(0.5 * logvar)
    - Decoder: z → [Linear + LayerNorm + ReLU + Dropout]* → Reconstruction
    - Classifier: z → Linear(1)

    Losses:
    - Reconstruction: MSELoss
    - KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
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

        # Camadas para mu e logvar
        self.fc_mu = nn.Linear(self.ENCODER_DIMS[1], self.LATENT_DIM)
        self.fc_logvar = nn.Linear(self.ENCODER_DIMS[1], self.LATENT_DIM)

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder: input → (mu, logvar).

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor]
            (mu, logvar) - parâmetros da distribuição latente.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma.

        Parâmetros
        ----------
        mu : torch.Tensor
            Média da distribuição latente.
        logvar : torch.Tensor
            Log da variância da distribuição latente.

        Retorna
        -------
        torch.Tensor
            Amostra do espaço latente z.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        # Modo de avaliação (model.eval()): retorna apenas a média
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder: latent → reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parâmetros
        ----------
        x : torch.Tensor
            Input (batch_size, input_dim).

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (logits, reconstruction, mu, logvar)
            - logits: (batch_size,) para classificação
            - reconstruction: (batch_size, input_dim)
            - mu: (batch_size, latent_dim)
            - logvar: (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        # A cabeça supervisionada usa mu para evitar que o ruído da
        # reparameterização degrade a classificação.
        logits = self.classifier(mu).squeeze(1)
        # A reconstrução continua estocástica no treino e determinística na
        # avaliação.
        z = self.reparameterize(mu, logvar) if self.training else mu
        reconstruction = self.decode(z)
        return logits, reconstruction, mu, logvar

    @torch.no_grad()
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna a representação latente (média mu).

        Para VAE, usamos mu (média) como representação determinística.

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
        mu, _ = self.encode(x)
        return mu
