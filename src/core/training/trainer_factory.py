"""
Factory para criação de trainers específicos por tipo de modelo.
"""
import torch.nn as nn
from torch.optim import Optimizer

from .encoder_trainer import EncoderTrainer
from .autoencoder_trainer import AutoencoderTrainer
from .vae_trainer import VAETrainer
from .encoder_fusion_inside_trainer import EncoderFusionInsideTrainer
from .autoencoder_fusion_inside_trainer import AutoencoderFusionInsideTrainer
from .vae_fusion_inside_trainer import VAEFusionInsideTrainer
from ..config.config import DEVICE, LAMBDA_CLASS, LAMBDA_RECON, BETA
from ..models.encoder import Encoder
from ..models.autoencoder import Autoencoder
from ..models.vae import VAE
from ..models.encoder_fusion_inside import EncoderFusionInside
from ..models.autoencoder_fusion_inside import AutoencoderFusionInside
from ..models.vae_fusion_inside import VAEFusionInside


def create_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    device=DEVICE,
    lambda_class: float = None,
    lambda_recon: float = None,
    beta: float = None,
) -> EncoderTrainer | AutoencoderTrainer | VAETrainer:
    """
    Cria o trainer apropriado baseado no tipo do modelo.

    Parâmetros
    ----------
    model : nn.Module
        Modelo a ser treinado.
    optimizer : Optimizer
        Otimizador configurado.
    device : torch.device
        Device (CPU/GPU).
    lambda_class : float, optional
        Peso da loss de classificação (usa LAMBDA_CLASS do config se None).
    lambda_recon : float, optional
        Peso da loss de reconstrução (usa LAMBDA_RECON do config se None).
    beta : float, optional
        Peso da KL divergence para VAE (usa BETA do config se None).

    Retorna
    -------
    EncoderTrainer | AutoencoderTrainer | VAETrainer | EncoderFusionInsideTrainer | AutoencoderFusionInsideTrainer
        Trainer apropriado para o tipo de modelo.
    """
    # Usa valores do config se não especificado
    if lambda_class is None:
        lambda_class = LAMBDA_CLASS
    if lambda_recon is None:
        lambda_recon = LAMBDA_RECON
    if beta is None:
        beta = BETA

    if isinstance(model, Encoder):
        # Encoder: apenas classificação
        criterion = nn.BCEWithLogitsLoss()
        return EncoderTrainer(model, criterion, optimizer, device)

    if isinstance(model, Autoencoder):
        # Autoencoder: dual loss (classification + reconstruction)
        criterion_class = nn.BCEWithLogitsLoss()
        criterion_recon = nn.MSELoss()
        return AutoencoderTrainer(
            model, criterion_class, criterion_recon, optimizer,
            lambda_class=lambda_class, lambda_recon=lambda_recon, device=device
        )

    if isinstance(model, VAE):
        # VAE: triple loss (classification + reconstruction + KL divergence)
        criterion_class = nn.BCEWithLogitsLoss()
        criterion_recon = nn.MSELoss()
        return VAETrainer(
            model, criterion_class, criterion_recon, optimizer,
            lambda_class=lambda_class, lambda_recon=lambda_recon, beta=beta, device=device
        )

    if isinstance(model, EncoderFusionInside):
        criterion = nn.BCEWithLogitsLoss()
        return EncoderFusionInsideTrainer(model, criterion, optimizer, device)

    if isinstance(model, AutoencoderFusionInside):
        criterion_class = nn.BCEWithLogitsLoss()
        criterion_recon = nn.MSELoss()
        return AutoencoderFusionInsideTrainer(
            model, criterion_class, criterion_recon, optimizer,
            lambda_class=lambda_class, lambda_recon=lambda_recon, device=device
        )

    if isinstance(model, VAEFusionInside):
        criterion_class = nn.BCEWithLogitsLoss()
        criterion_recon = nn.MSELoss()
        return VAEFusionInsideTrainer(
            model, criterion_class, criterion_recon, optimizer,
            lambda_class=lambda_class, lambda_recon=lambda_recon, beta=beta, device=device
        )

    raise ValueError(f"Unknown model type: {type(model).__name__}")
