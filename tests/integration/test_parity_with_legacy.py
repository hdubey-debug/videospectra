"""Integration parity: vnvideo end-to-end vs the legacy reference math.

Phase 6 already verifies that ``SpectralAnalyzer.update`` matches
``legacy_compute_entropy`` at 1e-9. This file extends the gate to the
full session pipeline: an embedding stream piped into a vnvideo
Session must emit FrameMetrics with values matching the legacy
reference for the same stream.

The legacy reference does not exist as a Session — it's the math
exported in ``tests/fixtures/legacy_compute_entropy.py``. We feed the
same synthetic embeddings into both and compare the per-frame numbers.
"""
from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
import pytest
from PIL import Image

from tests.fixtures.legacy_compute_entropy import compute_entropy as legacy_compute_entropy
from vnvideo._internal.normalize import l2_normalize
from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import ImageEmbedder
from vnvideo.events import FrameMetrics
from vnvideo.session import Session
from vnvideo.sinks import MemorySink
from vnvideo.types import Frame


def _random_normalized(n: int, d: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    arr = rng.randn(n, d)
    return l2_normalize(arr)


class _PinnedEmbedder:
    """An ImageEmbedder whose output is a fixed pre-computed sequence.

    Lets us drive the Session deterministically without rendering real
    images. The frame_id is used to index into the pinned sequence.
    """

    def __init__(self, sequence: np.ndarray) -> None:
        self._sequence = sequence

    def __call__(self, frames: list[Frame]) -> np.ndarray:
        # Use frame_id to index; one Frame per call in practice
        out = np.stack([self._sequence[f.frame_id] for f in frames], axis=0)
        return out


@pytest.mark.asyncio
async def test_full_session_metrics_match_legacy() -> None:
    """End-to-end Session emits FrameMetrics whose entropy/motion/anomaly
    agree with the legacy reference for the same embedding sequence."""

    cfg = SpectralConfig()
    n_frames = 60
    d = 64
    embeddings = _random_normalized(n_frames, d=d, seed=7)

    # Drive the Session via a pinned embedder
    pinned = _PinnedEmbedder(embeddings)
    img_emb = ImageEmbedder(
        embed_dim=d,
        space_id="parity/pinned@64/test",
        embed_fn=pinned,
    )

    sink = MemorySink()
    session = Session(
        frame_embedder=img_emb,
        sinks=[sink],
        spectral_config=cfg,
    )
    await session.start()

    for i in range(n_frames):
        await session.process_frame(
            Frame.from_pil(Image.new("RGB", (4, 4)), source_id="parity", frame_id=i)
        )

    await asyncio.sleep(0.1)
    await session.aclose()
    await sink.aclose()

    new_metrics: list[FrameMetrics] = []
    async for ev in sink:
        if isinstance(ev, FrameMetrics):
            new_metrics.append(ev)

    assert len(new_metrics) == n_frames

    # Run the legacy reference with the same input sequence
    legacy_buffer: deque = deque(maxlen=cfg.window_frames)
    prev_eigvecs = None
    prev_raw_eigvecs = None
    prev_full_eigvals = None
    prev_full_eigvecs = None
    prev_X = None
    legacy_rows: list[dict] = []
    for emb in embeddings:
        legacy_buffer.append(emb)
        res, ema_v, raw_v, X, eigvals, eigvecs = legacy_compute_entropy(
            legacy_buffer,
            prev_eigvecs,
            prev_raw_eigvecs,
            cfg.top_k_eigen,
            cfg.top_k_vec,
            prev_full_eigvals,
            prev_full_eigvecs,
            prev_X,
        )
        legacy_rows.append(res)
        prev_eigvecs = ema_v
        prev_raw_eigvecs = raw_v
        prev_X = X
        prev_full_eigvals = eigvals
        prev_full_eigvecs = eigvecs

    for i, (new_ev, legacy) in enumerate(zip(new_metrics, legacy_rows)):
        assert new_ev.payload.entropy_norm == pytest.approx(
            legacy["entropy_norm"], abs=1e-9
        ), f"frame {i}: entropy_norm drift"
        assert new_ev.payload.motion_score == pytest.approx(
            legacy["motion_score"], abs=1e-9
        ), f"frame {i}: motion_score drift"
        assert new_ev.payload.anomaly_score == pytest.approx(
            legacy["anomaly_score"], abs=1e-9
        ), f"frame {i}: anomaly_score drift"
        assert new_ev.payload.buffer_fill == legacy["buffer_fill"]
        assert new_ev.payload.effective_rank == pytest.approx(
            legacy["effective_rank"], abs=1e-9
        )
