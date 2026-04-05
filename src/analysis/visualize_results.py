"""
Script para gerar visualizações a partir dos resultados salvos dos experimentos.

Uso:
    python src/analysis/visualize_results.py <model_name>

Exemplos:
    python src/analysis/visualize_results.py encoder
    python src/analysis/visualize_results.py autoencoder
"""
import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Adiciona o diretório raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from src.core.config.config import get_results_dir, LABELS
from src.core.visualization.plots import (
    plot_histories,
    plot_tsne_representation,
    plot_tsne_embeddings,
)


def load_experiment_results(model_name: str) -> dict:
    """
    Carrega resultados salvos dos experimentos.

    Parâmetros
    ----------
    model_name : str
        Nome do modelo ("encoder" ou "autoencoder").

    Retorna
    -------
    dict
        Dicionário com histories, representations e labels.
    """
    results_dir = get_results_dir(model_name)

    # Verifica se o diretório existe
    if not os.path.exists(results_dir):
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Please run the experiment first: python src/scripts/main_{model_name}.py"
        )

    # Carrega históricos
    history_path = os.path.join(results_dir, f"{model_name}_histories.pkl")
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"Histories file not found: {history_path}\n"
            f"Please run the experiment first."
        )

    with open(history_path, "rb") as f:
        histories = pickle.load(f)

    # Carrega representações
    repr_path = os.path.join(results_dir, f"{model_name}_representations.npz")
    if not os.path.exists(repr_path):
        raise FileNotFoundError(
            f"Representations file not found: {repr_path}\n"
            f"Please run the experiment first."
        )

    representations = dict(np.load(repr_path))

    # Carrega labels
    labels_path = os.path.join(results_dir, "labels.npz")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Please run the experiment first."
        )

    labels_data = np.load(labels_path)
    y = labels_data["y"]
    label_names = labels_data["label_names"]

    return {
        "histories": histories,
        "representations": representations,
        "y": y,
        "label_names": label_names,
    }


def main():
    """Carrega resultados e gera visualizações."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from saved experiment results."
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=[
            "encoder",
            "autoencoder",
            "vae",
            "encoder_fusion_inside",
            "autoencoder_fusion_inside",
            "vae_fusion_inside",
        ],
        help="Name of the model to visualize",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase1", "phase2", "all"],
        default="all",
        help="Which phase to visualize (default: all)",
    )

    args = parser.parse_args()
    model_name = args.model_name

    print("="*55)
    print(f"LOADING RESULTS FOR {model_name.upper()}")
    print("="*55)

    # Carrega resultados
    data = load_experiment_results(model_name)
    histories = data["histories"]
    representations = data["representations"]
    y = data["y"]
    label_names = data["label_names"]

    # Cria LabelEncoder
    le = LabelEncoder()
    le.classes_ = label_names

    # Separa históricos por fase
    histories_p1 = {k: v for k, v in histories.items() if "+" not in k}
    histories_p2 = {k: v for k, v in histories.items() if "+" in k and "|" not in k}
    histories_p3 = {k: v for k, v in histories.items() if "|" in k}

    # Separa representações por fase
    repr_p1 = {k: v for k, v in representations.items() if "+" not in k}
    repr_p2 = {k: v for k, v in representations.items() if "+" in k and "|" not in k}

    # ========================================================================
    # VISUALIZAÇÕES
    # ========================================================================
    print("\n" + "="*55)
    print("GENERATING VISUALIZATIONS")
    print("="*55)

    # Plot de históricos (apenas fase 2, que tem combinações interessantes)
    if histories_p2 and (args.phase == "all" or args.phase == "phase2"):
        print("\nPlotting training histories (Phase 2)...")
        plot_histories(histories_p2, model_name=model_name.capitalize())

    # t-SNE das representações (fase 1 + fase 2 + phase3 forms)
    if args.phase == "all" or args.phase == "phase1":
        results_dir = get_results_dir(model_name)
        tsne_path = os.path.join(results_dir, f"{model_name}_tsne_embeddings.npz")
        if os.path.exists(tsne_path):
            print("\nLoading precomputed t-SNE embeddings...")
            tsne_embeddings = dict(np.load(tsne_path))
            plot_tsne_embeddings(tsne_embeddings, y, le, model_name=model_name.capitalize())
        else:
            print("\nGenerating t-SNE visualizations...")
            all_repr = {**repr_p1, **repr_p2}
            # include phase3 representation forms if present
            repr_p3 = {k: v for k, v in representations.items() if "|" in k}
            all_repr.update(repr_p3)
            plot_tsne_representation(all_repr, y, le, model_name=model_name.capitalize())

    print("\n" + "="*55)
    print("VISUALIZATIONS COMPLETED!")
    print("="*55)
    print(f"Figures saved at: {get_results_dir(model_name)}")


if __name__ == "__main__":
    main()
