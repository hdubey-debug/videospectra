"""Verbatim copy of compute_entropy from von-neumann-dashboard/server.py:177-306.

This file is the source of truth for the parity test in
``tests/test_spectral.py`` — it MUST NOT be modified. If the upstream
file at /mmfs1/.../von-neumann-dashboard/server.py changes, regenerate
this copy from the current source-of-truth lines 177-306.

Original file SHA at extraction (last verified 2026-05-27):
  /mmfs1/scratch/jacks.local/hdubey/02-video-embeddings/rzenembed/
    von-neumann-dashboard/server.py
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

EMA_ALPHA = 0.3


def compute_entropy(
    buffer: deque,
    prev_eigvecs: Optional[List[np.ndarray]],
    prev_raw_eigvecs: Optional[List[np.ndarray]],
    top_k_eigen: int,
    top_k_vec: int,
    prev_full_eigvals: Optional[np.ndarray] = None,
    prev_full_eigvecs: Optional[np.ndarray] = None,
    prev_X: Optional[np.ndarray] = None,
):
    """
    Compute von Neumann entropy + spectral features from a buffer of embeddings.
    Returns (result_dict, ema_eigvecs, raw_eigvecs, X, eigvals, eigvecs).
    """
    n = len(buffer)
    X = np.stack(list(buffer))          # (n, d)
    S = X @ X.T                         # Gram matrix (n x n)
    rho = S / n                         # density matrix, Tr(rho) = 1

    eigvals, eigvecs = np.linalg.eigh(rho)  # ascending order

    # Numerical stability: clip negatives, re-normalise
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total > 0:
        eigvals = eigvals / total

    # Von Neumann entropy
    mask = eigvals > 1e-15
    H = -np.sum(eigvals[mask] * np.log(eigvals[mask]))
    H_norm = H / np.log(n) if n > 1 else 0.0

    # Effective rank: exp(H) = number of active visual modes
    effective_rank = float(np.exp(H)) if H > 0 else 1.0

    # Top-k eigenvalues (descending)
    top_eigvals = eigvals[::-1][:top_k_eigen].tolist()

    # Raw eigenvectors (before EMA) — needed for subspace angle
    raw_vecs = []
    for i in range(min(top_k_vec, n)):
        col_idx = n - 1 - i
        raw_vecs.append(eigvecs[:, col_idx].copy())

    # Spectral leverage → motion score (how much the newest frame disturbs modes)
    k_leverage = 0.0
    for i in range(min(top_k_vec, n)):
        col_idx = n - 1 - i
        lam_i = eigvals[col_idx]
        v_ij = eigvecs[n - 1, col_idx]  # newest frame's component
        k_leverage += lam_i * v_ij ** 2

    top_k_energy = sum(eigvals[n - 1 - i] for i in range(min(top_k_vec, n)))
    if top_k_energy > 1e-15:
        motion_score = float(max(0.0, min(1.0, 1.0 - n * k_leverage / top_k_energy)))
    else:
        motion_score = 0.0

    # Subspace-fit anomaly: how well does the new frame fit the PREVIOUS eigenbasis?
    if prev_full_eigvecs is not None and prev_X is not None:
        x_new = X[-1]                          # newest embedding
        s = prev_X @ x_new                     # (n_prev,) similarity vector
        U_prev = prev_full_eigvecs             # (n_prev, n_prev) orthonormal
        lam_prev = prev_full_eigvals           # ascending order
        p = U_prev.T @ s                       # projection into prev eigenbasis
        p2 = p ** 2
        lam_max = float(lam_prev[-1]) + 1e-15
        weighted_sum = float(np.dot(p2, lam_prev))      # Σ p_i² · λ_i
        total_sq = float(np.dot(s, s)) * lam_max + 1e-15
        spectral_fit = weighted_sum / total_sq
        anomaly_score = float(np.clip(1.0 - spectral_fit, 0.0, 1.0))
    else:
        spectral_fit = 1.0
        anomaly_score = 0.0

    # Subspace angle (using raw vecs vs prev_raw_eigvecs)
    if (prev_raw_eigvecs is not None
            and len(prev_raw_eigvecs) > 0
            and len(raw_vecs) > 0
            and len(prev_raw_eigvecs[0]) == len(raw_vecs[0])):
        V_prev = np.column_stack(prev_raw_eigvecs)  # (n, k)
        V_curr = np.column_stack(raw_vecs)           # (n, k)
        M = V_prev.T @ V_curr                        # (k, k)
        _, sigmas, _ = np.linalg.svd(M)
        cos_theta = np.clip(sigmas[-1], -1, 1)       # smallest singular value
        subspace_angle = float(np.degrees(np.arccos(cos_theta)))
    else:
        subspace_angle = None  # buffer still filling, sizes don't match

    # Fiedler clusters (v2 sign pattern)
    if n >= 2:
        fiedler = raw_vecs[1] if len(raw_vecs) > 1 else raw_vecs[0]
        clusters = [1 if x >= 0 else 0 for x in fiedler]
    else:
        clusters = [0] * n

    # EMA-smoothed eigenvectors (for display, sign-aligned)
    top_vecs_np = []
    for i in range(min(top_k_vec, n)):
        vec = raw_vecs[i].copy()
        if (prev_eigvecs is not None
                and i < len(prev_eigvecs)
                and len(vec) == len(prev_eigvecs[i])):
            dot = np.dot(vec, prev_eigvecs[i])
            if dot < 0:
                vec = -vec
            vec = EMA_ALPHA * vec + (1 - EMA_ALPHA) * prev_eigvecs[i]
            nrm = np.linalg.norm(vec)
            if nrm > 1e-12:
                vec /= nrm
        top_vecs_np.append(vec)

    top_vecs = [v.tolist() for v in top_vecs_np]

    result = {
        "entropy_norm": float(H_norm),
        "entropy": float(H),
        "effective_rank": effective_rank,
        "n": n,
        "eigenvalues": top_eigvals,
        "top_eigenvectors": top_vecs,
        "fiedler_clusters": clusters,
        "subspace_angle": subspace_angle,
        "motion_score": motion_score,
        "anomaly_score": anomaly_score,
        "spectral_fit": float(spectral_fit),
        "is_anomaly": anomaly_score > 0.5,
        "buffer_fill": n,
    }
    return result, top_vecs_np, raw_vecs, X.copy(), eigvals.copy(), eigvecs.copy()
