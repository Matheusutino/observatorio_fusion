"""
Centralized project configuration.
"""
import os
import random
import numpy as np
import torch


# ===========================================================================
# PATHS
# ===========================================================================

DATA_PATH = "dataset/leandl_oesnpg_dados.parquet"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_PATH = os.path.join(
    EMBEDDINGS_DIR, "emb_prod=titulo_resumo_abstract_tema=nome_keywords.npz"
)
BASE_RESULTS_DIR = "results"


def get_results_dir(model_name: str) -> str:
    """
    Returns the results directory for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g. "encoder", "autoencoder").

    Returns
    -------
    str
        Path to the results directory (e.g. "results/encoder").
    """
    return os.path.join(BASE_RESULTS_DIR, model_name.lower())


# ===========================================================================
# EMBEDDING MODEL
# ===========================================================================

SBERT_MODEL = "ibm-granite/granite-embedding-278m-multilingual"


# ===========================================================================
# LABELS AND RANDOM STATE
# ===========================================================================

RANDOM_STATE = 42
LABELS = ["ALTA", "BAIXA"]


# ===========================================================================
# ENCODER ARCHITECTURE
# ===========================================================================

HIDDEN_DIMS = [512, 256, 128]  # hidden layers
DROPOUT = 0.3


# ===========================================================================
# AUTOENCODER AND VAE ARCHITECTURE
# ===========================================================================

LATENT_DIM = 128  # latent space dimension

# Loss weights (independent)
LAMBDA_CLASS = 1.0  # classification loss weight
LAMBDA_RECON = 1.0  # reconstruction loss weight
BETA = 1.0  # KL divergence weight (VAE only)


# ===========================================================================
# TRAINING HYPERPARAMETERS
# ===========================================================================

BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 30  # epochs without improvement in validation F1-macro


# ===========================================================================
# VALIDATION
# ===========================================================================

N_FOLDS = 5  # number of folds for stratified cross-validation


# ===========================================================================
# FUSION CONFIGURATION
# ===========================================================================

# Number of top operators to combine in Phase 2
TOP_K = 3

# Text source configurations for ablation (Phase 3)
SOURCE_CONFIGS = [
    ("titulo", "nome"),
    ("titulo", "nome_keywords"),
    ("resumo", "nome"),
    ("resumo", "nome_keywords"),
    ("abstract", "nome"),
    ("abstract", "nome_keywords"),
    ("keywords", "nome"),
    ("keywords", "nome_keywords"),
    ("resumo_abstract", "nome"),
    ("resumo_abstract", "nome_keywords"),
    ("titulo_resumo_abstract", "nome"),
    ("titulo_resumo_abstract", "nome_keywords"),  # baseline
]


# ===========================================================================
# SEEDS AND DEVICE INITIALIZATION
# ===========================================================================

def setup_environment():
    """Initialize seeds and configure the device."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.use_deterministic_algorithms(mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


DEVICE = setup_environment()
