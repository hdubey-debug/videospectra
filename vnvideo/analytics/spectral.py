"""Spectral analytics — von Neumann entropy, motion, anomaly, shots.

This module is the byte-equivalent port of the original
``compute_entropy`` from ``von-neumann-dashboard/server.py`` lines
177-306, wrapped in a stateful ``SpectralAnalyzer`` class. The math is
deliberately unchanged so the parity test in ``tests/test_spectral.py``
can verify identical metrics against the source of truth.

The class owns:

- A sliding embedding buffer (``deque(maxlen=window_frames)``)
- Rising-edge state for shot detection
- EMA-smoothed top eigenvectors (for sign-stable display)
- Previous full eigendecomposition (for subspace-fit anomaly)

Pure numpy. No torch, no GPU, no I/O.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SpectralConfig:
    """Sliding-window spectral analyzer configuration.

    Defaults match the von-neumann-dashboard demo exactly.
    """

    window_frames: int = 30
    """Sliding window length, in frames. At 2 FPS, 30 frames = 15 s."""

    top_k_eigen: int = 30
    """How many eigenvalues to report (top-K, descending)."""

    top_k_vec: int = 3
    """How many top eigenvectors to track for motion."""

    anomaly_threshold: float = 0.5
    """Above this, the frame is anomalous; rising edge = shot boundary."""

    ema_alpha: float = 0.3
    """EMA weight on the latest eigenvectors (vs previous smoothed)."""


@dataclass
class SpectralUpdate:
    """One frame's spectral metrics — returned from ``SpectralAnalyzer.update``."""

    entropy_norm: float
    """von Neumann entropy normalized by log(n). 0 = pure state, 1 = uniform."""

    motion_score: float
    """1 - spectral leverage of newest frame on top-k modes. 0..1."""

    anomaly_score: float
    """Subspace-fit residual vs previous eigenbasis. 0..1."""

    is_anomaly: bool
    """``anomaly_score > config.anomaly_threshold``."""

    is_shot_boundary: bool
    """Rising edge of ``is_anomaly`` — a NEW anomaly run starting now."""

    shot_count: int
    """Cumulative count of rising-edge transitions since reset."""

    buffer_fill: int
    """Number of frames currently in the sliding window (0..window_frames)."""

    infer_ms_spectral: float
    """Wall time for the spectral computation, milliseconds."""


# ---------------------------------------------------------------------------
# Verbatim port of compute_entropy from von-neumann-dashboard/server.py
# ---------------------------------------------------------------------------


def _spectral_step(
    buffer: deque,
    prev_eigvecs: list[np.ndarray] | None,
    prev_raw_eigvecs: list[np.ndarray] | None,
    top_k_eigen: int,
    top_k_vec: int,
    ema_alpha: float,
    prev_full_eigvals: np.ndarray | None = None,
    prev_full_eigvecs: np.ndarray | None = None,
    prev_X: np.ndarray | None = None,
) -> tuple[dict, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim port of compute_entropy from server.py:177-306.

    Returns ``(result_dict, ema_eigvecs, raw_eigvecs, X, eigvals, eigvecs)``.
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
    if prev_full_eigvecs is not None and prev_X is not None and prev_full_eigvals is not None:
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
            vec = ema_alpha * vec + (1 - ema_alpha) * prev_eigvecs[i]
            nrm = np.linalg.norm(vec)
            if nrm > 1e-12:
                vec /= nrm
        top_vecs_np.append(vec)

    top_vecs = [v.tolist() for v in top_vecs_np]

    result = {
        "entropy_norm": float(H_norm),
        "entropy": float(H),
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


# ---------------------------------------------------------------------------
# SpectralAnalyzer — stateful wrapper
# ---------------------------------------------------------------------------


@dataclass
class _SpectralState:
    """Mutable internal state — not exposed to users."""

    embedding_buffer: deque[np.ndarray] = field(default_factory=deque)
    prev_eigvecs: list[np.ndarray] | None = None
    prev_raw_eigvecs: list[np.ndarray] | None = None
    prev_full_eigvals: np.ndarray | None = None
    prev_full_eigvecs: np.ndarray | None = None
    prev_X: np.ndarray | None = None
    shot_count: int = 0
    prev_anomaly: float = 0.0


class SpectralAnalyzer:
    """Sliding-window spectral analytics over a stream of embeddings.

    Call :meth:`update` once per new L2-normalized embedding; receive
    a :class:`SpectralUpdate` with entropy/motion/anomaly/shot
    boundary metrics.

    Pure numpy — no torch, no GPU. Safe to construct one per Session;
    state lives entirely on the instance.
    """

    def __init__(self, config: SpectralConfig | None = None) -> None:
        self.config = config or SpectralConfig()
        self._state = _SpectralState(
            embedding_buffer=deque(maxlen=self.config.window_frames),
        )

    def reset(self) -> None:
        """Clear all state — new session, same config."""
        self._state = _SpectralState(
            embedding_buffer=deque(maxlen=self.config.window_frames),
        )

    def update(self, embedding: np.ndarray) -> SpectralUpdate:
        """Process one new embedding and return the spectral metrics."""
        cfg = self.config
        st = self._state

        t_start = time.perf_counter()

        st.embedding_buffer.append(np.asarray(embedding, dtype=np.float64))

        result, ema_vecs, raw_vecs, X_full, eigvals_full, eigvecs_full = _spectral_step(
            st.embedding_buffer,
            st.prev_eigvecs,
            st.prev_raw_eigvecs,
            cfg.top_k_eigen,
            cfg.top_k_vec,
            cfg.ema_alpha,
            st.prev_full_eigvals,
            st.prev_full_eigvecs,
            st.prev_X,
        )

        # Shot boundary — rising edge on anomaly_score > threshold
        cur_anom = float(result["anomaly_score"])
        is_shot = cur_anom > cfg.anomaly_threshold and st.prev_anomaly <= cfg.anomaly_threshold
        if is_shot:
            st.shot_count += 1

        # Persist state for next call
        st.prev_eigvecs = ema_vecs
        st.prev_raw_eigvecs = raw_vecs
        st.prev_X = X_full
        st.prev_full_eigvals = eigvals_full
        st.prev_full_eigvecs = eigvecs_full
        st.prev_anomaly = cur_anom

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return SpectralUpdate(
            entropy_norm=float(result["entropy_norm"]),
            motion_score=float(result["motion_score"]),
            anomaly_score=cur_anom,
            is_anomaly=bool(result["is_anomaly"]),
            is_shot_boundary=is_shot,
            shot_count=st.shot_count,
            buffer_fill=int(result["buffer_fill"]),
            infer_ms_spectral=elapsed_ms,
        )
