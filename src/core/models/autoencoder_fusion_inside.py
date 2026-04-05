"""
Modelo Autoencoder com fusão interna ao encoder.

Nesta arquitetura, as duas entradas (emb_prod e emb_tema) passam por branches
independentes antes de serem fundidas. A fusão acontece dentro do encoder,
permitindo que o modelo aprenda transformações específicas para cada entrada
antes de combiná-las.

Arquitetura:
- branch1: x1 → h1 (transformação aprendida para emb_prod)
- branch2: x2 → h2 (transformação aprendida para emb_tema)
- merge: h1, h2 → h (operação de fusão)
- encoder_shared: h → z (espaço latente)
- decoder: z → (x1_hat, x2_hat) (reconstrução separada)
- classifier: z → logit (classificação)
"""
import torch
import torch.nn as nn

from .base import BaseModel
from ..fusion.fusion import FUSION_OPS


class AutoencoderFusionInside(BaseModel):
    """
    Autoencoder com fusão interna ao encoder.

    Cada entrada passa por uma branch independente antes da fusão.
    A fusão gera uma representação comum que é projetada para um único
    espaço latente. A partir dele, dois decoders independentes
    reconstróem as entradas.
    """

    ENCODER_DIMS = [256, 128]
    LATENT_DIM = 64

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        branch_dims: list[int],
        shared_enc_dims: list[int],
        latent_dim: int,
        dropout: float,
        fusion_ops: list[str],
    ):
        """
        Parâmetros
        ----------
        input_dim1 : int
            Dimensão da primeira entrada (emb_prod).
        input_dim2 : int
            Dimensão da segunda entrada (emb_tema).
        branch_dims : list[int]
            Dimensões das camadas das branches independentes.
        shared_enc_dims : list[int]
            Dimensões do encoder compartilhado (após fusão).
        latent_dim : int
            Dimensão do espaço latente.
        dropout : float
            Taxa de dropout.
        fusion_ops : list[str]
            Lista de operações de fusão a aplicar (ex: ["sum", "concat"]).
        """
        super().__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.fusion_ops = fusion_ops

        # Mantemos os parâmetros na assinatura para compatibilidade com o
        # restante do projeto, mas a topologia é fixa por decisão de desenho.
        _ = (branch_dims, shared_enc_dims, latent_dim)

        # ========== Encoder 1 (emb_prod) ==========
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim1, self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ========== Encoder 2 (emb_tema) ==========
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim2, self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Calcula dimensão após fusão
        h1_dim = self.ENCODER_DIMS[-1]
        h2_dim = self.ENCODER_DIMS[-1]
        fused_dim = self._compute_fused_dim(h1_dim, h2_dim)

        # ========== Espaço latente compartilhado ==========
        self.to_latent = nn.Linear(fused_dim, self.LATENT_DIM)

        # ========== Cabeças de reconstrução separadas ==========
        # Decoder para x1
        self.decoder1 = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[1], self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], input_dim1),
        )

        # Decoder para x2
        self.decoder2 = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.ENCODER_DIMS[1]),
            nn.LayerNorm(self.ENCODER_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[1], self.ENCODER_DIMS[0]),
            nn.LayerNorm(self.ENCODER_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ENCODER_DIMS[0], input_dim2),
        )

        # ========== Classifier ==========
        self.classifier = nn.Linear(self.LATENT_DIM, 1)

    def _compute_fused_dim(self, h1_dim: int, h2_dim: int) -> int:
        """Calcula dimensão resultante da fusão."""
        dummy_h1 = torch.zeros(1, h1_dim)
        dummy_h2 = torch.zeros(1, h2_dim)
        fused = self._apply_fusion(dummy_h1, dummy_h2)
        return fused.shape[1]

    def _apply_fusion(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Aplica operações de fusão em h1 e h2."""
        parts = [FUSION_OPS[op](h1, h2) for op in self.fusion_ops]
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def encode(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Encoder: (x1, x2) → latent representation."""
        # Branches independentes
        h1 = self.branch1(x1)
        h2 = self.branch2(x2)

        # Fusão
        h = self._apply_fusion(h1, h2)

        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder: latent → (reconstruction1, reconstruction2)."""
        recon1 = self.decoder1(z)
        recon2 = self.decoder2(z)
        return recon1, recon2

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parâmetros
        ----------
        x1 : torch.Tensor
            Primeira entrada (batch_size, input_dim1).
        x2 : torch.Tensor
            Segunda entrada (batch_size, input_dim2).

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (logits, reconstruction1, reconstruction2)
            - logits: (batch_size,) para classificação
            - reconstruction1: (batch_size, input_dim1)
            - reconstruction2: (batch_size, input_dim2)
        """
        z = self.encode(x1, x2)
        logits = self.classifier(z).squeeze(1)
        recon1, recon2 = self.decode(z)
        return logits, recon1, recon2

    @torch.no_grad()
    def get_representation(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Retorna a representação latente (espaço latente do encoder).

        Parâmetros
        ----------
        x1 : torch.Tensor
            Primeira entrada (batch_size, input_dim1).
        x2 : torch.Tensor
            Segunda entrada (batch_size, input_dim2).

        Retorna
        -------
        torch.Tensor
            Representação latente (batch_size, latent_dim).
        """
        self.eval()
        return self.encode(x1, x2)
