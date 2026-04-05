"""
Lógica das fases experimentais para Autoencoder com fusão interna.

Este módulo implementa os experimentos de fusão interna especificamente para
modelos AutoencoderFusionInside.
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
from ..fusion.fusion import FUSION_OPS
from ..data.loader import get_embeddings_for_config
from .base_experiment import (
    build_result_row,
    extend_fold_details,
    run_cv_experiment,
    save_phase_outputs,
)


def run_fusion_inside_experiment(
    op_names: list[str],
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    lambda_class: float = 1.0,
    lambda_recon: float = 1.0,
) -> dict:
    """
    Treina e avalia um encoder com fusão interna usando validação cruzada.
    """
    label = "+".join(op_names)
    X1 = emb_prod.astype(np.float32)
    X2 = emb_tema.astype(np.float32)

    def make_loader(idx: np.ndarray, shuffle: bool, generator: torch.Generator | None):
        return DataLoader(
            TensorDataset(
                torch.tensor(X1[idx]),
                torch.tensor(X2[idx]),
                torch.tensor(y[idx]),
            ),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    def build_representation(model) -> np.ndarray:
        return model.get_representation(
            torch.tensor(X1).to(DEVICE),
            torch.tensor(X2).to(DEVICE),
        ).cpu().numpy()

    return run_cv_experiment(
        label=label,
        y=y,
        model_builder=lambda: model_class(
            input_dim1=X1.shape[1],
            input_dim2=X2.shape[1],
            fusion_ops=op_names,
            **model_kwargs,
        ),
        loader_builder=make_loader,
        representation_builder=build_representation,
        prediction_extractor=lambda eval_result: eval_result[-2:],
        input_shape_printer=lambda: f"  Input dim1: {X1.shape[1]}, dim2: {X2.shape[1]}",
        train_verbose=True,
    )


def run_phase1_fusion_inside(
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    model_name: str = "encoder_fusion_inside",
    lambda_class: float = 1.0,
    lambda_recon: float = 1.0,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fase 1: Testa operadores de fusão isolados."""
    print("\n" + "=" * 55)
    print("PHASE 1 — ISOLATED OPERATORS")
    print("=" * 55)

    results = []
    fold_details = []
    histories = {}
    representations = {}

    for op_name in FUSION_OPS:
        res = run_fusion_inside_experiment(
            [op_name], emb_prod, emb_tema, y,
            model_class, model_kwargs, le,
            lambda_class, lambda_recon
        )
        results.append(build_result_row(res))
        extend_fold_details(fold_details, res, fusion=res["fusion"])
        histories[res["fusion"]] = res["histories"][-1]
        representations[res["fusion"]] = res["representation"]
    df = save_phase_outputs(results, fold_details, model_name, "phase1")
    return df, histories, representations


def run_phase2_fusion_inside(
    emb_prod: np.ndarray,
    emb_tema: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    top_ops: list[str],
    model_name: str = "encoder_fusion_inside",
    lambda_class: float = 1.0,
    lambda_recon: float = 1.0,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fase 2: Testa combinações dos top-K operadores."""
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
        res = run_fusion_inside_experiment(
            list(combo), emb_prod, emb_tema, y,
            model_class, model_kwargs, le,
            lambda_class, lambda_recon
        )
        results.append(build_result_row(res))
        extend_fold_details(fold_details, res, fusion=res["fusion"])
        histories[res["fusion"]] = res["histories"][-1]
        representations[res["fusion"]] = res["representation"]
    df = save_phase_outputs(results, fold_details, model_name, "phase2")
    return df, histories, representations


def run_phase3_fusion_inside(
    df_data: pd.DataFrame,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    le: LabelEncoder,
    best_ops: list[str],
    model_name: str = "encoder_fusion_inside",
    lambda_class: float = 1.0,
    lambda_recon: float = 1.0,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fase 3: Ablação de fontes textuais usando os melhores operadores."""
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

        res = run_fusion_inside_experiment(
            best_ops, ep, et, y,
            model_class, model_kwargs, le,
            lambda_class, lambda_recon
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
