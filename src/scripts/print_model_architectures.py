"""
Imprime as arquiteturas dos modelos AE/VAE usados no projeto.
"""
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from src.core.config.config import DROPOUT, HIDDEN_DIMS, LATENT_DIM
from src.core.models.autoencoder import Autoencoder
from src.core.models.autoencoder_fusion_inside import AutoencoderFusionInside
from src.core.models.vae import VAE
from src.core.models.vae_fusion_inside import VAEFusionInside


INPUT_DIM = 768
BRANCH_DIMS = [256, 128]
SHARED_ENC_DIMS = [256, 128]
FUSION_CONFIGS = {
    "sum": ["sum"],
    "concat": ["concat"],
}


def print_block(title: str, model) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(model)
    print()


def main() -> None:
    ae = Autoencoder(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    )
    vae = VAE(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    )

    print_block("Autoencoder", ae)
    print_block("VAE", vae)

    for fusion_name, fusion_ops in FUSION_CONFIGS.items():
        ae_inside = AutoencoderFusionInside(
            input_dim1=INPUT_DIM,
            input_dim2=INPUT_DIM,
            branch_dims=BRANCH_DIMS,
            shared_enc_dims=SHARED_ENC_DIMS,
            latent_dim=LATENT_DIM,
            dropout=DROPOUT,
            fusion_ops=fusion_ops,
        )
        vae_inside = VAEFusionInside(
            input_dim1=INPUT_DIM,
            input_dim2=INPUT_DIM,
            branch_dims=BRANCH_DIMS,
            shared_enc_dims=SHARED_ENC_DIMS,
            latent_dim=LATENT_DIM,
            dropout=DROPOUT,
            fusion_ops=fusion_ops,
        )

        print_block(f"AutoencoderFusionInside [{fusion_name}]", ae_inside)
        print_block(f"VAEFusionInside [{fusion_name}]", vae_inside)


if __name__ == "__main__":
    main()
