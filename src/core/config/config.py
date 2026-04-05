"""
Configurações centralizadas do projeto.
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
    Retorna o diretório de resultados específico para um modelo.

    Parâmetros
    ----------
    model_name : str
        Nome do modelo (ex: "encoder", "autoencoder").

    Retorna
    -------
    str
        Caminho para o diretório de resultados (ex: "results/encoder").
    """
    return os.path.join(BASE_RESULTS_DIR, model_name.lower())


# ===========================================================================
# MODELO DE EMBEDDINGS
# ===========================================================================

SBERT_MODEL = "ibm-granite/granite-embedding-278m-multilingual"


# ===========================================================================
# RÓTULOS E RANDOM STATE
# ===========================================================================

RANDOM_STATE = 42
LABELS = ["ALTA", "BAIXA"]


# ===========================================================================
# ARQUITETURA DO ENCODER
# ===========================================================================

HIDDEN_DIMS = [512, 256, 128]  # camadas ocultas
DROPOUT = 0.3


# ===========================================================================
# ARQUITETURA DO AUTOENCODER E VAE
# ===========================================================================

LATENT_DIM = 128  # dimensão do espaço latente

# Pesos das losses (independentes)
LAMBDA_CLASS = 1.0  # peso da loss de classificação
LAMBDA_RECON = 1.0  # peso da loss de reconstrução
BETA = 1.0  # peso da KL divergence (apenas para VAE)


# ===========================================================================
# HIPERPARÂMETROS DE TREINO
# ===========================================================================

BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 30  # épocas sem melhora no F1-macro val


# ===========================================================================
# VALIDAÇÃO
# ===========================================================================

N_FOLDS = 5  # Número de folds para validação cruzada estratificada


# ===========================================================================
# CONFIGURAÇÕES DE FUSÃO
# ===========================================================================

# Número de melhores operadores a combinar na Fase 2
TOP_K = 3

# Configurações de fonte textual para ablação (Fase 3)
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
    ("titulo_resumo_abstract", "nome_keywords"),  # linha de base
]


# ===========================================================================
# INICIALIZAÇÃO DE SEEDS E DEVICE
# ===========================================================================

def setup_environment():
    """Inicializa seeds e configura device."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.use_deterministic_algorithms(mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


DEVICE = setup_environment()
