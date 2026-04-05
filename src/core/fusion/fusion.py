"""
Operadores de fusão de embeddings e funções auxiliares.
"""
import numpy as np
import torch


def _is_torch_tensor(x) -> bool:
    """Indica se a entrada é um tensor do PyTorch."""
    return isinstance(x, torch.Tensor)


def _concat(parts):
    """Concatena arrays/tensores preservando o backend."""
    first = parts[0]
    if _is_torch_tensor(first):
        return torch.cat(parts, dim=1)
    return np.hstack(parts)


def _geo_mean(u, v):
    """Média geométrica elemento a elemento."""
    p = u * v
    if _is_torch_tensor(p):
        return torch.sign(p) * torch.abs(p).pow(0.5)
    return np.sign(p) * np.abs(p) ** 0.5


def _v_ortho(u, v):
    """Componente ortogonal de v em relação a u."""
    if _is_torch_tensor(u):
        dot = (u * v).sum(dim=1, keepdim=True)
    else:
        dot = (u * v).sum(axis=1, keepdims=True)
    return v - dot * u


def _sigmoid_weighted(u, v, beta=5.0):
    """Ponderação sigmóide baseada na similaridade de cosseno."""
    if _is_torch_tensor(u):
        cos = (u * v).sum(dim=1, keepdim=True)
        alpha = torch.sigmoid(beta * cos)
    else:
        cos = (u * v).sum(axis=1, keepdims=True)
        alpha = 1.0 / (1.0 + np.exp(-beta * cos))
    return alpha * u + (1.0 - alpha) * v


# Dicionário com todos os operadores de fusão disponíveis
FUSION_OPS = {
    "sum": lambda u, v: u + v,
    "average": lambda u, v: (u + v) / 2,
    "concat": lambda u, v: _concat([u, v]),
    "diff": lambda u, v: torch.abs(u - v) if _is_torch_tensor(u) else np.abs(u - v),
    "hadamard": lambda u, v: u * v,
    "max": lambda u, v: torch.maximum(u, v) if _is_torch_tensor(u) else np.maximum(u, v),
    "min": lambda u, v: torch.minimum(u, v) if _is_torch_tensor(u) else np.minimum(u, v),
    "geo_mean": _geo_mean,
    "v_ortho": _v_ortho,
    "sigmoid_weighted": _sigmoid_weighted,
}


def apply_fusion(emb_prod: np.ndarray, emb_tema: np.ndarray, op_names: list[str]) -> np.ndarray:
    """
    Aplica um ou mais operadores de fusão e concatena os resultados.

    Parâmetros
    ----------
    emb_prod : np.ndarray
        Embeddings da produção (N, D).
    emb_tema : np.ndarray
        Embeddings do tema (N, D).
    op_names : list[str]
        Lista de nomes de operadores a serem aplicados.

    Retorna
    -------
    np.ndarray
        Embedding fusionado. Se múltiplos operadores, concatena os resultados.
    """
    parts = [FUSION_OPS[op](emb_prod, emb_tema) for op in op_names]
    return _concat(parts) if len(parts) > 1 else parts[0]
