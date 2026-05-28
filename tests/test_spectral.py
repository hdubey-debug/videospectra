"""Tests for videospectra.analytics.spectral — parity, property, performance."""
from __future__ import annotations

import pathlib
import re
import time
from collections import deque

import numpy as np
import pytest

from tests.fixtures.legacy_compute_entropy import compute_entropy as legacy_compute_entropy
from videospectra._internal.normalize import l2_normalize
from videospectra.analytics.spectral import (
    SpectralAnalyzer,
    SpectralConfig,
    SpectralUpdate,
    _spectral_step,
)


def _random_embeddings(n: int, d: int = 64, seed: int = 0) -> np.ndarray:
    """Deterministic L2-normalized embeddings."""
    rng = np.random.RandomState(seed)
    arr = rng.randn(n, d)
    return l2_normalize(arr)


# ---------------------------------------------------------------------------
# Parity test — the gate
# ---------------------------------------------------------------------------


class TestParityVsLegacy:
    """The new SpectralAnalyzer must produce byte-equivalent metrics
    to the legacy compute_entropy function on the same input sequence."""

    def test_parity_30_frames(self) -> None:
        cfg = SpectralConfig()  # defaults: 30 / 30 / 3 / 0.5 / 0.3
        embeddings = _random_embeddings(n=cfg.window_frames, d=64, seed=42)

        analyzer = SpectralAnalyzer(cfg)
        new_results: list[SpectralUpdate] = []
        for emb in embeddings:
            new_results.append(analyzer.update(emb))

        # Run the legacy function with the same state-passing pattern
        legacy_buffer: deque = deque(maxlen=cfg.window_frames)
        prev_eigvecs = None
        prev_raw_eigvecs = None
        prev_full_eigvals = None
        prev_full_eigvecs = None
        prev_X = None

        legacy_results: list[dict] = []
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
            legacy_results.append(res)
            prev_eigvecs = ema_v
            prev_raw_eigvecs = raw_v
            prev_X = X
            prev_full_eigvals = eigvals
            prev_full_eigvecs = eigvecs

        for i, (new, legacy) in enumerate(zip(new_results, legacy_results)):
            assert new.entropy_norm == pytest.approx(legacy["entropy_norm"], abs=1e-9), (
                f"frame {i}: entropy_norm drift"
            )
            assert new.motion_score == pytest.approx(legacy["motion_score"], abs=1e-9), (
                f"frame {i}: motion_score drift"
            )
            assert new.anomaly_score == pytest.approx(legacy["anomaly_score"], abs=1e-9), (
                f"frame {i}: anomaly_score drift"
            )
            assert new.is_anomaly == legacy["is_anomaly"]
            assert new.buffer_fill == legacy["buffer_fill"]

    def test_parity_60_frames_with_jump(self) -> None:
        """Inject a sudden change at frame 35 and verify both pipelines
        register an anomaly with the same magnitude."""
        cfg = SpectralConfig()
        first = _random_embeddings(n=35, d=64, seed=1)
        jumped = _random_embeddings(n=25, d=64, seed=2) + 3.0
        jumped = l2_normalize(jumped)
        embeddings = np.vstack([first, jumped])

        analyzer = SpectralAnalyzer(cfg)
        new_anom = []
        for emb in embeddings:
            new_anom.append(analyzer.update(emb).anomaly_score)

        legacy_buffer: deque = deque(maxlen=cfg.window_frames)
        prev_eigvecs = None
        prev_raw_eigvecs = None
        prev_full_eigvals = None
        prev_full_eigvecs = None
        prev_X = None
        legacy_anom = []
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
            legacy_anom.append(res["anomaly_score"])
            prev_eigvecs = ema_v
            prev_raw_eigvecs = raw_v
            prev_X = X
            prev_full_eigvals = eigvals
            prev_full_eigvecs = eigvecs

        for i, (n_v, l_v) in enumerate(zip(new_anom, legacy_anom)):
            assert n_v == pytest.approx(l_v, abs=1e-9), f"frame {i}: anomaly drift"


# ---------------------------------------------------------------------------
# Property tests — math behaves as expected on synthetic signals
# ---------------------------------------------------------------------------


class TestProperties:
    def test_constant_embeddings_entropy_zero(self) -> None:
        cfg = SpectralConfig()
        v = np.zeros(64)
        v[0] = 1.0
        analyzer = SpectralAnalyzer(cfg)
        for _ in range(cfg.window_frames):
            u = analyzer.update(v)
        # All embeddings identical -> rank 1 density matrix -> entropy 0
        assert u.entropy_norm == pytest.approx(0.0, abs=1e-9)

    def test_orthonormal_embeddings_entropy_one(self) -> None:
        # n orthonormal vectors in d-dim with d >= n give a uniform
        # spectrum, hence maximal von Neumann entropy = log(n), so
        # entropy_norm = 1.
        cfg = SpectralConfig(window_frames=10, top_k_eigen=10, top_k_vec=3)
        n = cfg.window_frames
        d = 64
        # First n columns of identity (each is unit-norm and orthogonal)
        embeddings = np.eye(d)[:n]
        analyzer = SpectralAnalyzer(cfg)
        for emb in embeddings:
            u = analyzer.update(emb)
        assert u.entropy_norm == pytest.approx(1.0, abs=1e-9)

    def test_shot_boundary_rising_edge(self) -> None:
        # Steady state, then sudden change at frame 35 -> shot.
        cfg = SpectralConfig()
        first = _random_embeddings(n=35, d=64, seed=1)
        jumped = l2_normalize(_random_embeddings(n=25, d=64, seed=2) + 3.0)
        embeddings = np.vstack([first, jumped])

        analyzer = SpectralAnalyzer(cfg)
        shot_count = 0
        for emb in embeddings:
            u = analyzer.update(emb)
            if u.is_shot_boundary:
                shot_count += 1

        # At least one shot boundary fires during the jump period.
        assert shot_count >= 1

    def test_shot_boundary_only_fires_on_rising_edge(self) -> None:
        # A sustained run above the anomaly threshold should never fire
        # the rising-edge shot boundary on the same frame twice. Verify
        # by directly driving prev_anomaly state via the update() loop.
        cfg = SpectralConfig(anomaly_threshold=0.5)
        analyzer = SpectralAnalyzer(cfg)
        # Establish a calm baseline (rank-1 — entropy 0, anomaly 0).
        v = np.zeros(64)
        v[0] = 1.0
        for _ in range(cfg.window_frames):
            analyzer.update(v)
        # Now flip to a completely orthogonal direction repeatedly. The
        # first frame in the new direction is anomalous (high subspace-
        # fit residual against the previous eigenbasis); subsequent
        # frames in the same direction are no longer rising edges.
        w = np.zeros(64)
        w[1] = 1.0
        shot_count = 0
        any_anomaly = False
        for _ in range(20):
            u = analyzer.update(w)
            if u.is_shot_boundary:
                shot_count += 1
            if u.is_anomaly:
                any_anomaly = True
        # Sanity: this contrived input *must* cross the anomaly
        # threshold somewhere — otherwise the test asserts nothing.
        assert any_anomaly, "test input did not produce any anomaly"
        # Even with many anomalous frames, the rising edge fires at most
        # once when the eigenbasis re-equilibrates. (Allow up to 3 in
        # case anomaly score dips below threshold briefly during the
        # window transition and then re-rises.)
        assert shot_count <= 3, f"shot boundary fired too often: {shot_count}"
        assert shot_count >= 1

# ---------------------------------------------------------------------------
# Memory & isolation
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_buffer_capped_at_window_frames(self) -> None:
        cfg = SpectralConfig(window_frames=10)
        analyzer = SpectralAnalyzer(cfg)
        embeddings = _random_embeddings(n=50, d=32, seed=0)
        for emb in embeddings:
            u = analyzer.update(emb)
        assert u.buffer_fill == 10
        assert len(analyzer._state.embedding_buffer) == 10

    def test_reset_clears_state(self) -> None:
        cfg = SpectralConfig()
        analyzer = SpectralAnalyzer(cfg)
        for emb in _random_embeddings(n=30, d=32, seed=0):
            analyzer.update(emb)
        assert analyzer._state.shot_count >= 0
        analyzer.reset()
        assert analyzer._state.shot_count == 0
        assert len(analyzer._state.embedding_buffer) == 0
        assert analyzer._state.prev_eigvecs is None

    def test_two_analyzers_independent(self) -> None:
        cfg = SpectralConfig()
        a = SpectralAnalyzer(cfg)
        b = SpectralAnalyzer(cfg)
        for emb in _random_embeddings(n=15, d=32, seed=0):
            a.update(emb)
        # a has state, b should not
        assert len(a._state.embedding_buffer) == 15
        assert len(b._state.embedding_buffer) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_one_update_under_5ms_on_cpu(self) -> None:
        cfg = SpectralConfig()  # 30 frames, k=3, k_eigen=30
        analyzer = SpectralAnalyzer(cfg)
        # Warm up the cache
        for emb in _random_embeddings(n=cfg.window_frames, d=64, seed=0):
            analyzer.update(emb)
        # Measure
        emb = _random_embeddings(n=1, d=64, seed=1)[0]
        # Run 20 iterations and take the median to suppress noise
        durations = []
        for _ in range(20):
            t0 = time.perf_counter()
            analyzer.update(emb)
            durations.append((time.perf_counter() - t0) * 1000.0)
        durations.sort()
        median = durations[len(durations) // 2]
        # 5ms is the spec target; allow 25ms to be robust to shared CI hosts
        assert median < 25.0, f"median spectral update too slow: {median:.2f} ms"


# ---------------------------------------------------------------------------
# Internal API stability — _spectral_step is the verbatim port
# ---------------------------------------------------------------------------


class TestVerbatimPort:
    def test_spectral_step_matches_legacy_one_frame(self) -> None:
        embeddings = _random_embeddings(n=5, d=16, seed=99)
        buf: deque = deque(maxlen=30)
        for emb in embeddings:
            buf.append(emb)

        new = _spectral_step(buf, None, None, 30, 3, 0.3)
        legacy = legacy_compute_entropy(buf, None, None, 30, 3)

        # The legacy fixture's result dict is a SUPERSET of new[0] (still
        # carries effective_rank — see fixture file, which is frozen).
        # The new result dict's keys must all appear in legacy's keys.
        assert set(new[0].keys()).issubset(set(legacy[0].keys()))
        for key in ("entropy_norm", "motion_score", "anomaly_score"):
            assert new[0][key] == pytest.approx(legacy[0][key], abs=1e-12)


class TestLegacyFixtureFreshness:
    """The fixture at tests/fixtures/legacy_compute_entropy.py is a manual
    copy of compute_entropy from von-neumann-dashboard/server.py:177-306.

    This test extracts the function body from the live source and
    compares it to the fixture. If the upstream file drifts, this
    fails with a clear instruction to regenerate the fixture — which
    is the only way the parity gate stays meaningful.
    """

    LIVE_SOURCE = (
        "/mmfs1/scratch/jacks.local/hdubey/02-video-embeddings/"
        "rzenembed/von-neumann-dashboard/server.py"
    )

    def test_fixture_matches_live_compute_entropy(self) -> None:
        live_path = pathlib.Path(self.LIVE_SOURCE)
        if not live_path.exists():
            pytest.skip(f"live source not present at {self.LIVE_SOURCE}")

        live_text = live_path.read_text()
        # Extract just the compute_entropy function body. We use a
        # naive marker pair that we control: the function begins at
        # 'def compute_entropy' and ends at the next top-level def or
        # '# ── '-style section divider.
        m = re.search(
            r"^def compute_entropy\b.*?(?=^\S|^# ?─)",
            live_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert m, "could not locate compute_entropy in live source"
        live_body = m.group(0).strip()

        fixture_path = pathlib.Path(__file__).parent / "fixtures" / "legacy_compute_entropy.py"
        fixture_text = fixture_path.read_text()
        m2 = re.search(
            r"^def compute_entropy\b.*?(?=^\S|^# ?─|\Z)",
            fixture_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert m2, "could not locate compute_entropy in fixture"
        fixture_body = m2.group(0).strip()

        # Normalize trailing whitespace
        live_body_norm = "\n".join(line.rstrip() for line in live_body.splitlines())
        fixture_body_norm = "\n".join(line.rstrip() for line in fixture_body.splitlines())

        assert live_body_norm == fixture_body_norm, (
            "tests/fixtures/legacy_compute_entropy.py has DRIFTED from the live "
            f"source at {self.LIVE_SOURCE}. Regenerate the fixture by copying "
            "the compute_entropy function verbatim. The parity gate is only "
            "meaningful when the fixture matches the source of truth."
        )
