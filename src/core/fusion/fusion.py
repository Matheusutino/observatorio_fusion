"""Embedding fusion operators and helper functions.
"""
import numpy as np
import torch


def _is_torch_tensor(x) -> bool:
    """Check whether the input is a PyTorch tensor.

    Args:
        x: Input object.

    Returns:
        True if x is a torch.Tensor, otherwise False.
    """
    return isinstance(x, torch.Tensor)


def _concat(parts):
    """Concatenate arrays or tensors preserving the backend.

    Args:
        parts: List of numpy arrays or torch tensors.

    Returns:
        Concatenated array or tensor along dimension 1.
    """
    first = parts[0]
    if _is_torch_tensor(first):
        return torch.cat(parts, dim=1)
    return np.hstack(parts)


def gated_symmetric(u, v):
    """Compute gated symmetric combination.

    Args:
        u: First tensor or array.
        v: Second tensor or array.

    Returns:
        Gated symmetric combination of u and v.
    """
    if _is_torch_tensor(u):
        gate_u = torch.sigmoid(u)
        gate_v = torch.sigmoid(v)
    else:
        gate_u = 1.0 / (1.0 + np.exp(-u))
        gate_v = 1.0 / (1.0 + np.exp(-v))

    return u * gate_v + v * gate_u


def _v_ortho(u, v):
    """Compute the component of v orthogonal to u.

    Args:
        u: First tensor or array.
        v: Second tensor or array.

    Returns:
        The orthogonal component of v relative to u.
    """
    if _is_torch_tensor(u):
        u_norm_sq = (u * u).sum(dim=1, keepdim=True) + 1e-8
        dot = (u * v).sum(dim=1, keepdim=True)
    else:
        u_norm_sq = (u * u).sum(axis=1, keepdims=True) + 1e-8
        dot = (u * v).sum(axis=1, keepdims=True)
    return v - (dot / u_norm_sq) * u


def _sigmoid_weighted(u, v, beta=5.0):
    """Compute a sigmoid-weighted combination of u and v.

    The weight is based on cosine similarity.

    Args:
        u: First tensor or array.
        v: Second tensor or array.
        beta: Sigmoid sharpness parameter.

    Returns:
        Weighted combination of u and v.
    """
    if _is_torch_tensor(u):
        u_norm = torch.sqrt((u * u).sum(dim=1, keepdim=True) + 1e-8)
        v_norm = torch.sqrt((v * v).sum(dim=1, keepdim=True) + 1e-8)
        dot = (u * v).sum(dim=1, keepdim=True)
        cos = dot / (u_norm * v_norm)
        alpha = torch.sigmoid(beta * cos)
    else:
        u_norm = np.sqrt((u * u).sum(axis=1, keepdims=True) + 1e-8)
        v_norm = np.sqrt((v * v).sum(axis=1, keepdims=True) + 1e-8)
        dot = (u * v).sum(axis=1, keepdims=True)
        cos = dot / (u_norm * v_norm)
        alpha = 1.0 / (1.0 + np.exp(-beta * cos))
    return alpha * u + (1.0 - alpha) * v


# Dictionary with all available fusion operators
FUSION_OPS = {
    "sum": lambda u, v: u + v,
    "average": lambda u, v: (u + v) / 2,
    "concat": lambda u, v: _concat([u, v]),
    "diff": lambda u, v: torch.abs(u - v) if _is_torch_tensor(u) else np.abs(u - v),
    "hadamard": lambda u, v: u * v,
    "max": lambda u, v: torch.maximum(u, v) if _is_torch_tensor(u) else np.maximum(u, v),
    "min": lambda u, v: torch.minimum(u, v) if _is_torch_tensor(u) else np.minimum(u, v),
    "gated_symmetric": gated_symmetric,
    "v_ortho": _v_ortho,
    "sigmoid_weighted": _sigmoid_weighted,
}


def apply_fusion(emb_prod: np.ndarray, emb_topic: np.ndarray, op_names: list[str]) -> np.ndarray:
    """Apply fusion operators and concatenate results.

    Args:
        emb_prod: Input embeddings for production (N, D).
    emb_topic: Input embeddings for topic (N, D).
        op_names: List of fusion operator names to apply.

    Returns:
        Fused embedding. If multiple operators are provided, the outputs are concatenated.
    """
    parts = [FUSION_OPS[op](emb_prod, emb_topic) for op in op_names]
    return _concat(parts) if len(parts) > 1 else parts[0]
