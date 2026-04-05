"""
Funções de cálculo de métricas e avaliação.
"""
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de classificação.

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    y_pred : np.ndarray
        Rótulos preditos (índices inteiros).

    Retorna
    -------
    dict
        Dicionário com f1_macro, accuracy, precision, recall.
    """
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder, labels: list[str]
) -> None:
    """
    Imprime o classification report detalhado.

    Parâmetros
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    y_pred : np.ndarray
        Rótulos preditos (índices inteiros).
    le : LabelEncoder
        LabelEncoder usado para transformar os rótulos.
    labels : list[str]
        Lista de nomes das classes.
    """
    print(
        classification_report(
            le.inverse_transform(y_true),
            le.inverse_transform(y_pred),
            labels=labels,
            zero_division=0,
        )
    )


def print_test_results(metrics: dict) -> None:
    """
    Imprime métricas de teste formatadas.

    Parâmetros
    ----------
    metrics : dict
        Dicionário com as métricas (f1_macro, accuracy, precision, recall).
    """
    print(f"\n  --- Teste ---")
    print(
        f"  F1-macro : {metrics['f1_macro']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}"
    )
