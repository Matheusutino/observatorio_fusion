"""Visualization utilities for training curves, t-SNE, and related plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import LabelEncoder

from ..config.config import RANDOM_STATE, get_results_dir


def save_tsne_embeddings(
    repr_dict: dict,
    model_name: str = "Encoder",
    n_neighbors: int = 15,
) -> str:
    """Compute and save precomputed UMAP embeddings.

    Args:
        repr_dict: Mapping from fusion or representation name to latent embeddings.
        model_name: Model name used for result path and file naming.
        n_neighbors: UMAP n_neighbors parameter.

    Returns:
        Path to the saved UMAP embeddings file.
    """
    umap_embeddings = {}
    for name, Z in repr_dict.items():
        print(f"  Precomputing UMAP for [{name}] ({Z.shape[0]} × {Z.shape[1]})...")
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=RANDOM_STATE, n_jobs=1)
        umap_embeddings[name] = reducer.fit_transform(Z)

    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{model_name.lower()}_umap_embeddings.npz")
    np.savez_compressed(save_path, **umap_embeddings)
    print(f"  Precomputed UMAP saved: {save_path}")
    return save_path


def plot_tsne_embeddings(
    tsne_dict: dict,
    y_all: np.ndarray,
    le: LabelEncoder,
    model_name: str = "Encoder",
) -> None:
    """Plot precomputed UMAP embeddings.

    Args:
        tsne_dict: Mapping from fusion or representation name to 2D UMAP embeddings.
        y_all: Label array for coloring points.
        le: LabelEncoder used to decode label indices.
        model_name: Model name used in plot titles and save paths.
    """
    # Translate labels
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
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
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
        f"UMAP — {model_name} Fusion Methods (complete dataset)",
        f"{model_name.lower()}_umap_fusion_methods.png",
        n_cols=5,
    )
    plot_grid(
        representation_forms,
        f"UMAP — {model_name} Representation Forms (complete dataset)",
        f"{model_name.lower()}_umap_representation_forms.png",
        n_cols=3,
    )


def plot_histories(
    histories: dict, model_name: str = "Encoder", window_size: int = 5, top_k: int = 3
) -> None:
    """Plot training and validation history curves for different fusion methods.

    This function selects the top-K best and bottom-K worst fusion variants
    and plots loss and validation F1 trajectories. For autoencoder models, it
    also plots separate classification and reconstruction losses.

    Args:
        histories: Mapping of fusion_name to training history dict.
        model_name: Model name used in titles and save paths.
        window_size: Smoothing window size for moving average.
        top_k: Number of best and worst fusion methods to include.
    """
    # Rank fusion methods by final validation F1
    ranked = sorted(
        histories.items(), key=lambda x: x[1]["val_f1"][-1], reverse=True
    )

    # Select top-K best and bottom-K worst methods
    selected = dict(ranked[:top_k] + ranked[-top_k:])

    # Detect whether this is an autoencoder model
    first_history = next(iter(selected.values()))
    is_autoencoder = "train_class_loss" in first_history

    # Configure subplots: 3 columns for autoencoder, otherwise 2
    n_cols = 3 if is_autoencoder else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))

    def smooth(values, window):
        """Apply a moving average smoothing filter."""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    # Collect handles and labels for a single legend
    handles, labels = [], []

    for fusion_name, h in selected.items():
        # Apply smoothing
        train_loss_smooth = smooth(h["train_loss"], window_size)
        val_loss_smooth = smooth(h["val_loss"], window_size)
        val_f1_smooth = smooth(h["val_f1"], window_size)

        # Adjust x-axis for smoothed values
        x_loss = np.arange(len(train_loss_smooth))
        x_f1 = np.arange(len(val_f1_smooth))

        axes[0].plot(x_loss, train_loss_smooth, linestyle="-", alpha=0.7)
        axes[0].plot(x_loss, val_loss_smooth, linestyle="--", alpha=0.7)

        # Plot on the second axis and collect handle for legend
        line, = axes[1].plot(x_f1, val_f1_smooth)
        handles.append(line)
        labels.append(fusion_name)

        # If autoencoder, plot individual loss components
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

    # Single horizontal legend below the plots
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
    """Plot latent embeddings via UMAP for fusion and representation forms.

    This function generates two separate figures:
    1. Fusion methods (sum, average, concat, diff, hadamard, max, min,
       geo_mean, v_ortho, sigmoid_weighted).
    2. Representation forms for researcher/topic pairings (prod|topic combinations).

    Args:
        repr_dict: Mapping of fusion_name to latent matrix Z of shape (N, repr_dim).
        y_all: True labels as integer indices.
        le: LabelEncoder used to decode label classes.
        model_name: Model name used in titles and save paths.
    """
    # Translate labels
    label_translation = {"ALTA": "High", "BAIXA": "Low"}
    colors = {"ALTA": "#2ecc71", "BAIXA": "#e74c3c"}

    # Split representations into two groups
    fusion_methods = {}
    representation_forms = {}

    for name, Z in repr_dict.items():
        if "|" in name:
            representation_forms[name] = Z
        else:
            fusion_methods[name] = Z

    # ========================================================================
    # FIGURE 1: Fusion Methods
    # ========================================================================
    if fusion_methods:
        n_fusion = len(fusion_methods)
        # Compute dynamic grid layout for fusion methods
        n_cols = 5
        n_rows = (n_fusion + n_cols - 1) // n_cols  # round up

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_fusion == 1 else axes

        for idx, (fusion_name, Z) in enumerate(fusion_methods.items()):
            ax = axes[idx]
            print(
                f"  UMAP representation [{fusion_name}] ({Z.shape[0]} × {Z.shape[1]})..."
            )
            reducer = UMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE, n_jobs=1)
            Z_2d = reducer.fit_transform(Z)

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
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(markerscale=2, fontsize=9)

        # Hide empty axes
        for idx in range(n_fusion, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"UMAP — {model_name} Fusion Methods (complete dataset)",
            fontsize=14,
        )
        plt.tight_layout()

        results_dir = get_results_dir(model_name)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{model_name.lower()}_umap_fusion_methods.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {save_path}")

    # ========================================================================
    # FIGURE 2: Researcher/Topic Representation Forms
    # ========================================================================
    if representation_forms:
        n_repr = len(representation_forms)
        # Compute dynamic grid layout for representation forms
        n_cols = 3
        n_rows = (n_repr + n_cols - 1) // n_cols  # round up

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_repr == 1 else axes

        for idx, (repr_name, Z) in enumerate(representation_forms.items()):
            ax = axes[idx]
            print(
                f"  UMAP representation [{repr_name}] ({Z.shape[0]} × {Z.shape[1]})..."
            )
            reducer = UMAP(n_components=2, n_neighbors=15, random_state=RANDOM_STATE, n_jobs=1)
            Z_2d = reducer.fit_transform(Z)

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
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(markerscale=2, fontsize=9)

        # Hide empty axes
        for idx in range(n_repr, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"UMAP — {model_name} Representation Forms (complete dataset)",
            fontsize=14,
        )
        plt.tight_layout()

        results_dir = get_results_dir(model_name)
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{model_name.lower()}_umap_representation_forms.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {save_path}")
