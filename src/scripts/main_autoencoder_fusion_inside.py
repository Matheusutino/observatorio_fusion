"""
Script principal para experimentos com Autoencoder com fusão interna.

Executa as 3 fases experimentais:
1. Operadores de fusão isolados
2. Combinações dos top-K operadores
3. Ablação de fontes textuais
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Adiciona o diretório raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.core.config.config import (
    RANDOM_STATE,
    LABELS,
    LATENT_DIM,
    DROPOUT,
    get_results_dir,
    TOP_K,
)
from src.core.data.loader import load_data, get_embeddings
from src.core.models.autoencoder_fusion_inside import AutoencoderFusionInside
from src.core.experiments.phases_autoencoder_fusion_inside import (
    run_phase1_fusion_inside,
    run_phase2_fusion_inside,
    run_phase3_fusion_inside,
)
from src.core.visualization.plots import save_tsne_embeddings


# ===========================================================================
# CONFIGURAÇÕES ESPECÍFICAS PARA FUSION INSIDE
# ===========================================================================

# Dimensões das branches independentes (antes da fusão)
BRANCH_DIMS = [256, 128]

# Dimensões do encoder compartilhado (após fusão)
SHARED_ENC_DIMS = [256, 128]

# Pesos das losses
LAMBDA_CLASS = 1.0  # peso da loss de classificação
LAMBDA_RECON = 1.0  # peso da loss de reconstrução


def main():
    """Executa o pipeline completo de experimentos com AutoencoderFusionInside."""
    model_name = "autoencoder_fusion_inside"
    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Carrega dados
    print("="*55)
    print("LOADING DATA")
    print("="*55)
    df = load_data()

    # Prepara labels
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(df["nivel_binario"])

    # Carrega embeddings padrão
    print("\n" + "="*55)
    print("LOADING EMBEDDINGS")
    print("="*55)
    emb_prod, emb_tema = get_embeddings(df)

    # Configuração do modelo
    model_kwargs = {
        "branch_dims": BRANCH_DIMS,
        "shared_enc_dims": SHARED_ENC_DIMS,
        "latent_dim": LATENT_DIM,
        "dropout": DROPOUT,
    }

    print("\n" + "="*55)
    print("MODEL CONFIGURATION")
    print("="*55)
    print(f"  Branch dims: {BRANCH_DIMS}")
    print(f"  Shared encoder dims: {SHARED_ENC_DIMS}")
    print(f"  Latent dim: {LATENT_DIM}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Lambda class: {LAMBDA_CLASS}")
    print(f"  Lambda recon: {LAMBDA_RECON}")

    # ========================================================================
    # FASE 1: Operadores isolados
    # ========================================================================
    df_phase1, histories_p1, repr_p1 = run_phase1_fusion_inside(
        emb_prod,
        emb_tema,
        y,
        AutoencoderFusionInside,
        model_kwargs,
        le,
        model_name=model_name,
        lambda_class=LAMBDA_CLASS,
        lambda_recon=LAMBDA_RECON,
    )

    # Extrai top-K operadores
    top_ops = df_phase1.head(TOP_K)["fusion"].tolist()
    print(f"\nTop-{TOP_K} operators: {top_ops}")

    # ========================================================================
    # FASE 2: Combinações dos top-K
    # ========================================================================
    df_phase2, histories_p2, repr_p2 = run_phase2_fusion_inside(
        emb_prod,
        emb_tema,
        y,
        AutoencoderFusionInside,
        model_kwargs,
        le,
        top_ops,
        model_name=model_name,
        lambda_class=LAMBDA_CLASS,
        lambda_recon=LAMBDA_RECON,
    )

    # Melhor configuração de operadores (fase 1 + fase 2)
    all_results = pd.concat([df_phase1, df_phase2]).sort_values("f1_macro_mean", ascending=False)
    best_ops = all_results.iloc[0]["fusion"].split("+")

    print(
        f"\nBest operator/combination: {'+'.join(best_ops)}  "
        f"F1={all_results.iloc[0]['f1_macro_mean']:.4f} ± {all_results.iloc[0]['f1_macro_std']:.4f}"
    )

    # ========================================================================
    # FASE 3: Ablação de fonte textual
    # ========================================================================
    _df_phase3, histories_p3, repr_p3 = run_phase3_fusion_inside(
        df,
        y,
        AutoencoderFusionInside,
        model_kwargs,
        le,
        best_ops,
        model_name=model_name,
        lambda_class=LAMBDA_CLASS,
        lambda_recon=LAMBDA_RECON,
    )

    # ========================================================================
    # SALVAMENTO DE RESULTADOS
    # ========================================================================
    print("\n" + "="*55)
    print("SAVING EXPERIMENT DATA")
    print("="*55)

    # Salva históricos de treino (pickle)
    import pickle
    all_histories = {**histories_p1, **histories_p2, **histories_p3}
    history_path = os.path.join(results_dir, f"{model_name}_histories.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(all_histories, f)
    print(f"  Histories saved: {history_path}")

    # Salva representações latentes (npz)
    all_repr = {**repr_p1, **repr_p2, **repr_p3}
    repr_path = os.path.join(results_dir, f"{model_name}_representations.npz")
    np.savez_compressed(repr_path, **all_repr)
    print(f"  Representations saved: {repr_path}")

    save_tsne_embeddings(all_repr, model_name=model_name)

    # Salva labels
    labels_path = os.path.join(results_dir, "labels.npz")
    np.savez_compressed(labels_path, y=y, label_names=le.classes_)
    print(f"  Labels saved: {labels_path}")

    print("\n" + "="*55)
    print("EXPERIMENTS COMPLETED!")
    print("="*55)
    print(f"Results saved at: {results_dir}")
    print("\nTo generate visualizations, run:")
    print(f"  python src/analysis/visualize_results.py {model_name}")


if __name__ == "__main__":
    main()
