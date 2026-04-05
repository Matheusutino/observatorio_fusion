"""
Funções de visualização (curvas de treino, t-SNE, etc).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from ..config.config import RANDOM_STATE, get_results_dir


def save_tsne_embeddings(
    repr_dict: dict,
    model_name: str = "Encoder",
    perplexity: int = 40,
) -> str:
    """
    Calcula e salva embeddings t-SNE pré-computadas para todas as representações.

    Retorna o caminho do arquivo salvo.
    """
    tsne_embeddings = {}
    for name, Z in repr_dict.items():
        print(f"  Precomputing t-SNE for [{name}] ({Z.shape[0]} × {Z.shape[1]})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
        tsne_embeddings[name] = tsne.fit_transform(Z)

    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{model_name.lower()}_tsne_embeddings.npz")
    np.savez_compressed(save_path, **tsne_embeddings)
    print(f"  Precomputed t-SNE saved: {save_path}")
    return save_path


def plot_tsne_embeddings(
    tsne_dict: dict,
    y_all: np.ndarray,
    le: LabelEncoder,
    model_name: str = "Encoder",
) -> None:
    """
    Plota representações t-SNE já pré-computadas.
    """
    # Tradução dos labels
    label_translation = {"ALTA": "High", "BAIXA": "Low"}
    colors = {"ALTA": "#2ecc71", "BAIXA": "#e74c3c"}

    fusion_methods = {}
    representation_forms = {}
    for name, Z in tsne_dict.items():
        if "|" in name:
            representation_forms[name] = Z
        else:
            fusion_methods[name] = Z

    def plot_grid(plot_dict: dict, title_fmt: str, filename: str, n_cols: int):
        if not plot_dict:
            return
        n_items = len(plot_dict)
        n_rows = (n_items + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_items == 1 else axes

        for idx, (name, Z_2d) in enumerate(plot_dict.items()):
            ax = axes[idx]
            for cls_idx, lbl in enumerate(le.classes_):
                mask = y_all == cls_idx
                ax.scatter(
                    Z_2d[mask, 0],
                    Z_2d[mask, 1],
                    c=colors[lbl],
                    s=12,
                    alpha=0.6,
                    linewidths=0,
                    label=label_translation[lbl],
                )
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(markerscale=2, fontsize=9)

        for idx in range(n_items, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title_fmt, fontsize=14)
        plt.tight_layout()
        results_dir = get_results_dir(model_name)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {save_path}")

    plot_grid(
        fusion_methods,
        f"t-SNE — {model_name} Fusion Methods (complete dataset)",
        f"{model_name.lower()}_tsne_fusion_methods.png",
        n_cols=5,
    )
    plot_grid(
        representation_forms,
        f"t-SNE — {model_name} Representation Forms (complete dataset)",
        f"{model_name.lower()}_tsne_representation_forms.png",
        n_cols=3,
    )


def plot_histories(
    histories: dict, model_name: str = "Encoder", window_size: int = 5, top_k: int = 3
) -> None:
    """
    Plota curvas de loss e F1-macro de validação para diferentes fusões.
    Mostra apenas top-K melhores e piores fusões, com média móvel aplicada.

    Para Autoencoder, plota também as losses individuais (classification e reconstruction).

    Parâmetros
    ----------
    histories : dict
        Dicionário {fusion_name: history}, onde history é {"train_loss", "val_loss", "val_f1"}.
        Para Autoencoder: também {"train_class_loss", "train_recon_loss", "val_class_loss", "val_recon_loss"}.
    model_name : str
        Nome do modelo para título (ex: "Encoder", "Autoencoder").
    window_size : int
        Tamanho da janela para média móvel (padrão: 5).
    top_k : int
        Número de melhores e piores fusões a plotar (padrão: 3).
    """
    # Ranqueia fusões pelo F1 final de validação
    ranked = sorted(
        histories.items(), key=lambda x: x[1]["val_f1"][-1], reverse=True
    )

    # Seleciona top-K e bottom-K
    selected = dict(ranked[:top_k] + ranked[-top_k:])

    # Detecta se é autoencoder
    first_history = next(iter(selected.values()))
    is_autoencoder = "train_class_loss" in first_history

    # Configura subplots: 3 colunas se autoencoder, 2 caso contrário
    n_cols = 3 if is_autoencoder else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))

    def smooth(values, window):
        """Aplica média móvel."""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    # Coleta handles e labels para legenda única
    handles, labels = [], []

    for fusion_name, h in selected.items():
        # Aplica suavização
        train_loss_smooth = smooth(h["train_loss"], window_size)
        val_loss_smooth = smooth(h["val_loss"], window_size)
        val_f1_smooth = smooth(h["val_f1"], window_size)

        # Ajusta eixo x para valores suavizados
        x_loss = np.arange(len(train_loss_smooth))
        x_f1 = np.arange(len(val_f1_smooth))

        axes[0].plot(x_loss, train_loss_smooth, linestyle="-", alpha=0.7)
        axes[0].plot(x_loss, val_loss_smooth, linestyle="--", alpha=0.7)

        # Plota no segundo gráfico e coleta handle para legenda
        line, = axes[1].plot(x_f1, val_f1_smooth)
        handles.append(line)
        labels.append(fusion_name)

        # Se for autoencoder, plota losses individuais
        if is_autoencoder:
            train_class_smooth = smooth(h["train_class_loss"], window_size)
            train_recon_smooth = smooth(h["train_recon_loss"], window_size)
            val_class_smooth = smooth(h["val_class_loss"], window_size)
            val_recon_smooth = smooth(h["val_recon_loss"], window_size)

            x_class_recon = np.arange(len(train_class_smooth))
            axes[2].plot(x_class_recon, train_class_smooth, linestyle="-", alpha=0.5)
            axes[2].plot(x_class_recon, train_recon_smooth, linestyle="-", alpha=0.5)
            axes[2].plot(x_class_recon, val_class_smooth, linestyle="--", alpha=0.5)
            axes[2].plot(x_class_recon, val_recon_smooth, linestyle="--", alpha=0.5)

    axes[0].set_title("Total Loss per Epoch (moving average)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")

    axes[1].set_title("Validation F1-Macro per Epoch (moving average)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-Macro")

    if is_autoencoder:
        axes[2].set_title("Individual Losses per Epoch (moving average)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Loss")
        axes[2].legend(["Train Class", "Train Recon", "Val Class", "Val Recon"], fontsize=7)

    plt.suptitle(
        f"{model_name} — Top {top_k} Best and Worst Fusions", fontsize=13
    )

    # Legenda horizontal única abaixo dos gráficos
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{model_name.lower()}_fusion_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {save_path}")


def plot_tsne_representation(
    repr_dict: dict,
    y_all: np.ndarray,
    le: LabelEncoder,
    model_name: str = "Encoder",
) -> None:
    """
    Para cada fusão, projeta a representação latente em 2D via t-SNE.
    Cria dois gráficos separados:
    1. Métodos de fusão (sum, average, concat, diff, hadamard, max, min, geo_mean, v_ortho, sigmoid_weighted)
    2. Formas de representação pesquisador/tema (combinações prod|tema)

    Parâmetros
    ----------
    repr_dict : dict
        Dicionário {fusion_name: Z}, onde Z é (N, repr_dim).
    y_all : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    le : LabelEncoder
        LabelEncoder usado para transformar os rótulos.
    model_name : str
        Nome do modelo para título.
    """
    # Tradução dos labels
    label_translation = {"ALTA": "High", "BAIXA": "Low"}
    colors = {"ALTA": "#2ecc71", "BAIXA": "#e74c3c"}

    # Separa representações em dois grupos
    fusion_methods = {}
    representation_forms = {}

    for name, Z in repr_dict.items():
        if "|" in name:
            representation_forms[name] = Z
        else:
            fusion_methods[name] = Z

    # ========================================================================
    # GRÁFICO 1: Métodos de Fusão
    # ========================================================================
    if fusion_methods:
        n_fusion = len(fusion_methods)
        # Calcula grid dinâmico para métodos de fusão
        n_cols = 5
        n_rows = (n_fusion + n_cols - 1) // n_cols  # arredonda para cima

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_fusion == 1 else axes

        for idx, (fusion_name, Z) in enumerate(fusion_methods.items()):
            ax = axes[idx]
            print(
                f"  t-SNE representation [{fusion_name}] ({Z.shape[0]} × {Z.shape[1]})..."
            )
            tsne = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE)
            Z_2d = tsne.fit_transform(Z)

            for cls_idx, lbl in enumerate(le.classes_):
                mask = y_all == cls_idx
                ax.scatter(
                    Z_2d[mask, 0],
                    Z_2d[mask, 1],
                    c=colors[lbl],
                    s=12,
                    alpha=0.6,
                    linewidths=0,
                    label=label_translation[lbl],
                )

            ax.set_title(f"Fusion: {fusion_name}", fontsize=12)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(markerscale=2, fontsize=9)

        # Esconde eixos vazios
        for idx in range(n_fusion, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"t-SNE — {model_name} Fusion Methods (complete dataset)",
            fontsize=14,
        )
        plt.tight_layout()

        results_dir = get_results_dir(model_name)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{model_name.lower()}_tsne_fusion_methods.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {save_path}")

    # ========================================================================
    # GRÁFICO 2: Formas de Representação Pesquisador/Tema
    # ========================================================================
    if representation_forms:
        n_repr = len(representation_forms)
        # Calcula grid dinâmico para formas de representação
        n_cols = 3
        n_rows = (n_repr + n_cols - 1) // n_cols  # arredonda para cima

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_repr == 1 else axes

        for idx, (repr_name, Z) in enumerate(representation_forms.items()):
            ax = axes[idx]
            print(
                f"  t-SNE representation [{repr_name}] ({Z.shape[0]} × {Z.shape[1]})..."
            )
            tsne = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE)
            Z_2d = tsne.fit_transform(Z)

            for cls_idx, lbl in enumerate(le.classes_):
                mask = y_all == cls_idx
                ax.scatter(
                    Z_2d[mask, 0],
                    Z_2d[mask, 1],
                    c=colors[lbl],
                    s=12,
                    alpha=0.6,
                    linewidths=0,
                    label=label_translation[lbl],
                )

            ax.set_title(f"{repr_name}", fontsize=11)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(markerscale=2, fontsize=9)

        # Esconde eixos vazios
        for idx in range(n_repr, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"t-SNE — {model_name} Representation Forms (complete dataset)",
            fontsize=14,
        )
        plt.tight_layout()

        results_dir = get_results_dir(model_name)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{model_name.lower()}_tsne_representation_forms.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {save_path}")
