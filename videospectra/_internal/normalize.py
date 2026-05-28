"""L2 normalization helper.

Applied at the Session boundary to every embedding the user's BYOM
callable returns, unconditionally. There is no opt-out — see invariant
5 in ``docs/architecture-v0.1.md``.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(arr: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize along the last axis.

    Accepts 1-D (single vector) or 2-D (batch of vectors). Returns a new
    float64 array; the input is not mutated.

    A vector whose norm is below ``eps`` is returned unchanged (still
    zero) rather than scaled by 1/0 — see the von Neumann pipeline,
    where a flat constant signal *should* map to a zero density matrix
    rather than blow up.
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        norm = float(np.linalg.norm(out))
        if norm < eps:
            return out.copy()
        return out / norm
    if out.ndim != 2:
        raise ValueError(f"l2_normalize expects 1-D or 2-D array, got shape {out.shape}")
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    safe_norms = np.where(norms < eps, 1.0, norms)
    return out / safe_norms
