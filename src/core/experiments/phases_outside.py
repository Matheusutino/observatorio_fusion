"""
Lógica das fases experimentais para modelos com fusão externa.

Este módulo contém a implementação real usada por Autoencoder, VAE e Encoder
quando a fusão é aplicada antes da entrada do modelo.
"""
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from ..config.config import (
    BATCH_SIZE,
    DEVICE,
    SOURCE_CONFIGS,
)
from ..fusion.fusion import FUSION_OPS, apply_fusion
from ..data.loader import get_embeddings_for_config
from .base_experiment import (
    build_result_row,
    extend_fold_details,
    run_cv_experiment,
    save_phase_outputs,
)


def run_fusion_experiment(
    op_names: list[str],
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
) -> dict:
    """
    Treina e avalia um modelo usando validação cruzada estratificada.
    """
    label = "+".join(op_names)
    X = apply_fusion(emb_prod, emb_tema, op_names).astype(np.float32)

    def make_loader(idx: np.ndarray, shuffle: bool, generator: torch.Generator | None):
        return DataLoader(
            TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx])),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def extract_predictions(eval_result: tuple) -> tuple[np.ndarray, np.ndarray]:
        return eval_result[-2:]

    def build_representation(model) -> np.ndarray:
        X_tensor = torch.tensor(X).to(DEVICE)
        if hasattr(model, "cond_dim"):
            y_tensor = torch.tensor(y).to(DEVICE)
            return model.get_representation(X_tensor, y_tensor).cpu().numpy()

        return model.get_representation(X_tensor).cpu().numpy()

    return run_cv_experiment(
        label=label,
        y=y,
        model_builder=lambda: model_class(input_dim=X.shape[1], **model_kwargs),
        loader_builder=make_loader,
        representation_builder=build_representation,
        prediction_extractor=extract_predictions,
        input_shape_printer=lambda: f"  Input dim: {X.shape[1]}",
        train_verbose=True,
    )


def run_phase1(
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    model_name: str = "encoder",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Fase 1: Testa operadores de fusão isolados.
    """
    print("\n" + "=" * 55)
    print("PHASE 1 — ISOLATED OPERATORS")
    print("=" * 55)

    results = []
    fold_details = []
    histories = {}
    representations = {}

    for op_name in FUSION_OPS:
        res = run_fusion_experiment(
            [op_name],
            emb_prod,
            emb_tema,
            y,
            model_class,
            model_kwargs,
            le,
        )
        results.append(build_result_row(res))
        extend_fold_details(fold_details, res, fusion=res["fusion"])
        histories[res["fusion"]] = res["histories"][-1]
        representations[res["fusion"]] = res["representation"]
    df = save_phase_outputs(results, fold_details, model_name, "phase1")
    return df, histories, representations


def run_phase2(
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    top_ops: list[str],
    model_name: str = "encoder",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Fase 2: Testa combinações dos top-K operadores.
    """
    print("\n" + "=" * 55)
    print("PHASE 2 — TOP-3 COMBINATIONS")
    print("=" * 55)

    combos = []
    for r in range(2, len(top_ops) + 1):
        combos += list(itertools.combinations(top_ops, r))

    results = []
    fold_details = []
    histories = {}
    representations = {}

    for combo in combos:
        res = run_fusion_experiment(
            list(combo),
            emb_prod,
            emb_tema,
            y,
            model_class,
            model_kwargs,
            le,
        )
        results.append(build_result_row(res))
        extend_fold_details(fold_details, res, fusion=res["fusion"])
        histories[res["fusion"]] = res["histories"][-1]
        representations[res["fusion"]] = res["representation"]
    df = save_phase_outputs(results, fold_details, model_name, "phase2")
    return df, histories, representations


def run_phase3(
    df_data: pd.DataFrame,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    best_ops: list[str],
    model_name: str = "encoder",
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Fase 3: Ablação de fontes textuais usando os melhores operadores.
    """
    print("\n" + "=" * 55)
    print("PHASE 3 — TEXT SOURCE ABLATION")
    print(f"  Operator(s): {'+'.join(best_ops)}")
    print("=" * 55)

    results = []
    fold_details = []
    histories = {}
    representations = {}

    for prod_cfg, tema_cfg in SOURCE_CONFIGS:
        config_label = f"prod={prod_cfg}|tema={tema_cfg}"
        print(f"\n  Config: {config_label}")

        ep, et = get_embeddings_for_config(df_data, prod_cfg, tema_cfg)

        res = run_fusion_experiment(
            best_ops,
            ep,
            et,
            y,
            model_class,
            model_kwargs,
            le,
        )
        results.append(build_result_row(res, prod_config=prod_cfg, tema_config=tema_cfg))
        extend_fold_details(fold_details, res, prod_config=prod_cfg, tema_config=tema_cfg)
        histories[config_label] = res["histories"][-1]
        representations[config_label] = res["representation"]
    df = save_phase_outputs(
        results,
        fold_details,
        model_name,
        "phase3",
        display_columns=[
            "prod_config",
            "tema_config",
            "f1_macro_mean",
            "f1_macro_std",
            "accuracy_mean",
            "accuracy_std",
        ],
    )
    return df, histories, representations
