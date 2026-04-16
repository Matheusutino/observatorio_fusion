"""
Núcleo compartilhado para execução de experimentos com validação cruzada.
"""
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config.config import (
    DEVICE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    INNER_VAL_RATIO,
    LR,
    N_FOLDS,
    RANDOM_STATE,
    WEIGHT_DECAY,
    get_results_dir,
)
from ..training.metrics import compute_metrics
from ..training.trainer_factory import create_trainer


MetricRow = dict[str, float | int]


def run_cv_experiment(
    label: str,
    y: np.ndarray,
    model_builder: Callable[[], torch.nn.Module],
    loader_builder: Callable[[np.ndarray, bool, torch.Generator | None], torch.utils.data.DataLoader],
    representation_builder: Callable[[torch.nn.Module], np.ndarray],
    prediction_extractor: Callable[[tuple], tuple[np.ndarray, np.ndarray]],
    input_shape_printer: Callable[[], str],
    trainer_kwargs: dict | None = None,
    train_verbose: bool = False,
) -> dict:
    """Executa um experimento com CV estratificada e validação interna."""
    print(f"\n{'=' * 55}")
    print(f"FUSION: {label} (CV with {N_FOLDS} folds)")
    print("=" * 55)
    print(input_shape_printer())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    idx_all = np.arange(len(y))

    fold_results: list[MetricRow] = []
    histories = []
    last_representation = None
    trainer_kwargs = trainer_kwargs or {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(idx_all, y)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

        train_inner_idx, val_inner_idx = train_test_split(
            train_idx,
            test_size=INNER_VAL_RATIO,
            shuffle=True,
            stratify=y[train_idx],
            random_state=RANDOM_STATE + fold_idx,
        )

        print(
            "    Split sizes: "
            f"train={len(train_inner_idx)} | "
            f"val={len(val_inner_idx)} | "
            f"test={len(test_idx)}"
        )

        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE + fold_idx)

        loader_train = loader_builder(train_inner_idx, True, generator)
        loader_val = loader_builder(val_inner_idx, False, None)
        loader_test = loader_builder(test_idx, False, None)

        model = model_builder().to(DEVICE)

        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params:,}")
            print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        trainer = create_trainer(model, optimizer, device=DEVICE, **trainer_kwargs)

        history = trainer.train(
            loader_train,
            loader_val,
            EPOCHS,
            EARLY_STOP_PATIENCE,
            verbose=train_verbose,
        )
        histories.append(history)

        y_true, y_pred = prediction_extractor(trainer.eval_epoch(loader_test))
        metrics = compute_metrics(y_true, y_pred)

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        print(f"    Test F1-Macro: {metrics['f1_macro']:.4f}")

        if fold_idx == N_FOLDS - 1:
            last_representation = representation_builder(model)

    df_folds = pd.DataFrame(fold_results)

    print("\n  CV Results:")
    print(f"    F1-Macro:  {df_folds['f1_macro'].mean():.4f} ± {df_folds['f1_macro'].std():.4f}")
    print(f"    Accuracy:  {df_folds['accuracy'].mean():.4f} ± {df_folds['accuracy'].std():.4f}")
    print(f"    Precision: {df_folds['precision'].mean():.4f} ± {df_folds['precision'].std():.4f}")
    print(f"    Recall:    {df_folds['recall'].mean():.4f} ± {df_folds['recall'].std():.4f}")

    return {
        "fusion": label,
        "f1_macro_mean": round(df_folds["f1_macro"].mean(), 4),
        "f1_macro_std": round(df_folds["f1_macro"].std(), 4),
        "accuracy_mean": round(df_folds["accuracy"].mean(), 4),
        "accuracy_std": round(df_folds["accuracy"].std(), 4),
        "precision_mean": round(df_folds["precision"].mean(), 4),
        "precision_std": round(df_folds["precision"].std(), 4),
        "recall_mean": round(df_folds["recall"].mean(), 4),
        "recall_std": round(df_folds["recall"].std(), 4),
        "fold_results": fold_results,
        "histories": histories,
        "representation": last_representation,
    }


def build_result_row(result: dict, **extra_fields) -> dict:
    """Monta a linha agregada de resultado para uma configuração."""
    row = {
        "fusion": result["fusion"],
        "f1_macro_mean": result["f1_macro_mean"],
        "f1_macro_std": result["f1_macro_std"],
        "accuracy_mean": result["accuracy_mean"],
        "accuracy_std": result["accuracy_std"],
        "precision_mean": result["precision_mean"],
        "precision_std": result["precision_std"],
        "recall_mean": result["recall_mean"],
        "recall_std": result["recall_std"],
    }
    row.update(extra_fields)
    return row


def extend_fold_details(fold_details: list[dict], result: dict, **extra_fields) -> None:
    """Acumula resultados por fold com contexto adicional."""
    for fold_result in result["fold_results"]:
        fold_details.append({**extra_fields, **fold_result})


def save_phase_outputs(
    results: list[dict],
    fold_details: list[dict],
    model_name: str,
    phase_name: str,
    display_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Ordena, imprime e salva os artefatos de uma fase."""
    df = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)

    print(f"\n--- {phase_name.replace('_', ' ').title()} Ranking ---")
    if display_columns is None:
        print(df.to_string(index=False))
    else:
        print(df[display_columns].to_string(index=False))

    results_dir = get_results_dir(model_name)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, f"{model_name}_{phase_name}_results.csv"), index=False)

    if fold_details:
        pd.DataFrame(fold_details).to_csv(
            os.path.join(results_dir, f"{model_name}_{phase_name}_folds.csv"),
            index=False,
        )

    return df
