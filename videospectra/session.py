"""Session orchestrator — the integration point.

Owns:
- frame buffer + spectral analyzer
- clip frame accumulator + clip task lifecycle
- text-prompt embeddings (for clip-level event scoring)
- embedder concurrency semaphore (default: 1)
- sink runners (one per registered Sink, with bounded queues)

Does NOT own:
- sockets, file I/O, frame sources (those live in sources/, server/)
- model integrations (those are user-supplied via the BYOM embedder
  wrappers)

The frame loop: ``await session.process_frame(frame)`` does spectral
analytics inline (fast, < 5ms on CPU) and kicks off clip embedding as
a fire-and-forget task when the accumulator fills. Every observable
state change is emitted as an Event via the registered sinks.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image as _PILImage

from videospectra._internal.normalize import l2_normalize
from videospectra.analytics.spectral import SpectralAnalyzer, SpectralConfig
from videospectra.embedders import ImageEmbedder, TextEmbedder, VideoEmbedder
from videospectra.events import (
    AnomalyAlert,
    AnomalyAlertPayload,
    ClipScore,
    ClipScores,
    ClipScoresPayload,
    EmbedderError,
    EmbedderErrorPayload,
    Event,
    FrameDropped,
    FrameDroppedPayload,
    FrameMetrics,
    FrameMetricsPayload,
    PromptAdded,
    PromptAddedPayload,
    PromptRemoved,
    PromptRemovedPayload,
    SessionInfo,
    SessionInfoPayload,
    ShotBoundary,
    ShotBoundaryPayload,
)
from videospectra.sinks import OverflowPolicy, Sink, SinkRunner
from videospectra.types import Capabilities, Frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipConfig:
    """Configuration for clip-level event detection.

    Seconds-based: ``clip_duration_seconds`` is the length of each clip
    in wall-clock seconds; ``clip_stride_seconds`` is the gap between
    successive clip starts. The Session converts these into integer
    frame counts using its ``source_fps``.

    Defaults match the von-neumann-dashboard demo: 4-second clips with
    no overlap (stride == duration). At the default 2 FPS, that resolves
    to 8 frames per clip.
    """

    clip_duration_seconds: float = 4.0
    clip_stride_seconds: float = 4.0
    max_events: int = 10


@dataclass(frozen=True)
class _ResolvedClipConfig:
    """Frame-count form of ClipConfig, derived from source_fps."""

    clip_frames: int
    clip_stride: int


def _resolve_clip_config(cfg: ClipConfig, source_fps: float) -> _ResolvedClipConfig:
    """Convert seconds-based ClipConfig into integer frame counts.

    Logs a warning if rounding shifts either field by more than 5%.
    """
    if cfg.clip_duration_seconds <= 0:
        raise ValueError(
            f"clip_duration_seconds must be > 0, got {cfg.clip_duration_seconds!r}"
        )
    if cfg.clip_stride_seconds <= 0:
        raise ValueError(
            f"clip_stride_seconds must be > 0, got {cfg.clip_stride_seconds!r}"
        )
    clip_frames = max(1, round(cfg.clip_duration_seconds * source_fps))
    clip_stride = max(1, round(cfg.clip_stride_seconds * source_fps))
    # Warn when rounding distorts the requested duration noticeably.
    actual_dur = clip_frames / source_fps
    actual_stride = clip_stride / source_fps
    for label, requested, actual in (
        ("clip_duration_seconds", cfg.clip_duration_seconds, actual_dur),
        ("clip_stride_seconds", cfg.clip_stride_seconds, actual_stride),
    ):
        if requested > 0 and abs(actual - requested) / requested > 0.05:
            logger.warning(
                "ClipConfig.%s requested %.3fs but rounds to %.3fs at source_fps=%.3f",
                label, requested, actual, source_fps,
            )
    return _ResolvedClipConfig(clip_frames=clip_frames, clip_stride=clip_stride)


class EmbedderValidationError(RuntimeError):
    """An embedder returned output that violates its contract.

    Carries the role and the expected vs actual mismatch.
    """

    def __init__(self, role: str, expected: str, actual: str) -> None:
        super().__init__(f"{role} embedder validation failed: expected {expected}, got {actual}")
        self.role = role
        self.expected = expected
        self.actual = actual


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _PromptRecord:
    prompt_id: str
    text: str
    embedding: np.ndarray  # L2-normalized


@dataclass
class _AnomalyTracker:
    """Tracks sustained anomaly runs to emit AnomalyAlert events at most
    once per crossing.

    ``min_sustained`` is the minimum consecutive frames above threshold
    before an alert fires (suppresses single-frame spikes).
    """

    threshold: float
    min_sustained: int = 3
    consecutive: int = 0
    fired_for_run: bool = False

    def step(self, score: float) -> tuple[bool, int]:
        """Return (should_fire, consecutive_count). Resets on dip."""
        if score > self.threshold:
            self.consecutive += 1
            if self.consecutive >= self.min_sustained and not self.fired_for_run:
                self.fired_for_run = True
                return True, self.consecutive
            return False, self.consecutive
        # Below threshold: reset
        self.consecutive = 0
        self.fired_for_run = False
        return False, 0


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Source-agnostic video understanding session.

    See ``docs/architecture-v0.1.md`` for the full contract. Briefly:
    construct with one or more BYOM embedders, optionally a spectral
    config and a clip config, and a list of sinks. Push frames in with
    ``await process_frame(frame)``; events come out via sinks.
    """

    def __init__(
        self,
        *,
        frame_embedder: ImageEmbedder | None = None,
        clip_embedder: VideoEmbedder | None = None,
        text_embedder: TextEmbedder | None = None,
        sinks: Sequence[Sink] = (),
        spectral_config: SpectralConfig | None = None,
        clip_config: ClipConfig | None = None,
        source_fps: float = 2.0,
        session_id: str | None = None,
        embedder_concurrency: int = 1,
        sink_max_queue: int = 100,
        sink_overflow: OverflowPolicy = "drop_oldest",
        anomaly_min_sustained: int = 3,
    ) -> None:
        self._validate_embedder_combination(frame_embedder, clip_embedder, text_embedder, clip_config)
        if source_fps <= 0:
            raise ValueError(f"source_fps must be > 0, got {source_fps!r}")

        self.frame_embedder = frame_embedder
        self.clip_embedder = clip_embedder
        self.text_embedder = text_embedder
        self.spectral_config = spectral_config
        self.clip_config = clip_config
        self.source_fps = float(source_fps)
        self._resolved_clip = (
            _resolve_clip_config(clip_config, self.source_fps) if clip_config is not None else None
        )
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.embedder_concurrency = max(1, embedder_concurrency)
        self.sink_max_queue = sink_max_queue
        self.sink_overflow: OverflowPolicy = sink_overflow

        self._spectral = (
            SpectralAnalyzer(spectral_config)
            if spectral_config is not None and frame_embedder is not None
            else None
        )
        self._anomaly_tracker = _AnomalyTracker(
            threshold=spectral_config.anomaly_threshold if spectral_config else 0.5,
            min_sustained=anomaly_min_sustained,
        )

        self._embed_sem = asyncio.Semaphore(self.embedder_concurrency)
        self._embed_executor = ThreadPoolExecutor(
            max_workers=self.embedder_concurrency,
            thread_name_prefix=f"videospectra-embed-{self.session_id}",
        )

        self._clip_accumulator: list[Frame] = []
        self._clip_pending = False
        self._clip_tasks: set[asyncio.Task] = set()

        self._prompts: dict[str, _PromptRecord] = {}
        self._prompts_lock = asyncio.Lock()

        self._sink_runners: list[SinkRunner] = []
        for sink in sinks:
            self._sink_runners.append(self._wrap_sink(sink))

        self._frames_processed = 0
        self._frames_dropped = 0
        self._in_flight = 0
        self._closing = False
        self._started = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_embedder_combination(
        frame_embedder: ImageEmbedder | None,
        clip_embedder: VideoEmbedder | None,
        text_embedder: TextEmbedder | None,
        clip_config: ClipConfig | None,
    ) -> None:
        if frame_embedder is None and clip_embedder is None:
            raise ValueError(
                "Session requires at least one of frame_embedder or clip_embedder"
            )
        # Clip and text are paired
        if clip_embedder is not None and text_embedder is None:
            raise ValueError(
                "clip_embedder requires a paired text_embedder (clip-level events "
                "score clips against text prompts)"
            )
        if text_embedder is not None and clip_embedder is None:
            raise ValueError(
                "text_embedder requires a paired clip_embedder (text alone has no "
                "consumer in the v0.1 pipeline)"
            )
        if clip_embedder is not None and text_embedder is not None:
            if clip_embedder.space_id != text_embedder.space_id:
                raise ValueError(
                    f"clip_embedder.space_id={clip_embedder.space_id!r} does not match "
                    f"text_embedder.space_id={text_embedder.space_id!r}; cosine similarity "
                    "across mismatched embedding spaces is meaningless"
                )
            if clip_embedder.embed_dim != text_embedder.embed_dim:
                raise ValueError(
                    f"clip_embedder.embed_dim={clip_embedder.embed_dim} does not match "
                    f"text_embedder.embed_dim={text_embedder.embed_dim}"
                )
        if clip_embedder is not None and clip_config is None:
            raise ValueError("clip_embedder requires a clip_config")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        for runner in self._sink_runners:
            runner.start()
        self._started = True
        await self._emit(self._make_session_info())

    async def aclose(self) -> None:
        self._closing = True
        # Wait for in-flight clip tasks (with a bounded timeout)
        if self._clip_tasks:
            await asyncio.wait(self._clip_tasks, timeout=5.0)
            for task in list(self._clip_tasks):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
        # Close sink runners (each drains its own queue)
        await asyncio.gather(
            *(runner.aclose() for runner in self._sink_runners),
            return_exceptions=True,
        )
        self._embed_executor.shutdown(wait=False)

    async def warmup(self) -> None:
        """Smoke-test every configured embedder with a tiny synthetic input.

        Raises EmbedderValidationError if any embedder returns the wrong
        shape, NaN/Inf values, or fails entirely. Intended to be called
        before accepting frames so a misconfigured setup fails loudly
        rather than per-frame.
        """
        sample_frame = Frame.from_pil(
            _PILImage.new("RGB", (4, 4), (128, 128, 128)),
            source_id="warmup",
            frame_id=0,
        )

        if self.frame_embedder is not None:
            _ = await self._call_embedder(self.frame_embedder, [sample_frame], role="frame")

        if self.clip_embedder is not None and self._resolved_clip is not None:
            sample_clip = [sample_frame] * self._resolved_clip.clip_frames
            _ = await self._call_embedder(self.clip_embedder, [sample_clip], role="clip")

        if self.text_embedder is not None:
            _ = await self._call_embedder(self.text_embedder, ["warmup"], role="text")

    # ------------------------------------------------------------------
    # Capabilities & SessionInfo
    # ------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        names: dict[str, str] = {}
        spaces: dict[str, str] = {}
        if self.frame_embedder is not None:
            names["frame"] = self.frame_embedder.name
            spaces["frame"] = self.frame_embedder.space_id
        if self.clip_embedder is not None:
            names["clip"] = self.clip_embedder.name
            spaces["clip"] = self.clip_embedder.space_id
        if self.text_embedder is not None:
            names["text"] = self.text_embedder.name
            spaces["text"] = self.text_embedder.space_id
        return Capabilities(
            spectral_analytics=self._spectral is not None,
            clip_events=self.clip_embedder is not None,
            prompts=self.text_embedder is not None,
            embedder_names=names,
            embedder_spaces=spaces,
        )

    def _make_session_info(self) -> SessionInfo:
        cfg: dict[str, Any] = {}
        if self.spectral_config is not None:
            cfg["window_frames"] = self.spectral_config.window_frames
            cfg["anomaly_threshold"] = self.spectral_config.anomaly_threshold
        if self.clip_config is not None and self._resolved_clip is not None:
            cfg["clip_duration_seconds"] = self.clip_config.clip_duration_seconds
            cfg["clip_stride_seconds"] = self.clip_config.clip_stride_seconds
            cfg["clip_frames"] = self._resolved_clip.clip_frames
            cfg["clip_stride"] = self._resolved_clip.clip_stride
        cfg["source_fps"] = self.source_fps
        cfg["embedder_concurrency"] = self.embedder_concurrency
        caps = self.capabilities()
        return SessionInfo(
            session_id=self.session_id,
            source_id=None,
            frame_id=None,
            monotonic_ts=time.perf_counter(),
            wall_ts=datetime.now(timezone.utc),
            payload=SessionInfoPayload(
                spectral_analytics=caps.spectral_analytics,
                clip_events=caps.clip_events,
                prompts=caps.prompts,
                embedder_names=caps.embedder_names,
                embedder_spaces=caps.embedder_spaces,
                config=cfg,
            ),
        )

    # ------------------------------------------------------------------
    # Sink management
    # ------------------------------------------------------------------

    def _wrap_sink(self, sink: Sink) -> SinkRunner:
        return SinkRunner(
            sink,
            max_queue=self.sink_max_queue,
            overflow=self.sink_overflow,
        )

    async def add_sink(self, sink: Sink) -> None:
        runner = self._wrap_sink(sink)
        self._sink_runners.append(runner)
        if self._started:
            runner.start()

    async def remove_sink(self, sink: Sink) -> None:
        for runner in list(self._sink_runners):
            if runner.sink is sink:
                self._sink_runners.remove(runner)
                await runner.aclose()
                return

    async def _emit(self, event: Event) -> None:
        for runner in self._sink_runners:
            await runner.enqueue(event)

    # ------------------------------------------------------------------
    # Prompts (text embeddings for clip-level event scoring)
    # ------------------------------------------------------------------

    async def add_prompt(self, text: str) -> str:
        if self.text_embedder is None:
            raise RuntimeError("Session has no text_embedder; cannot add prompts")
        prompt_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        emb = await self._call_embedder(self.text_embedder, [text], role="text")
        embed_ms = (time.perf_counter() - t0) * 1000.0
        async with self._prompts_lock:
            self._prompts[prompt_id] = _PromptRecord(
                prompt_id=prompt_id, text=text, embedding=emb[0]
            )
        await self._emit(
            PromptAdded(
                session_id=self.session_id,
                source_id=None,
                frame_id=None,
                monotonic_ts=time.perf_counter(),
                wall_ts=datetime.now(timezone.utc),
                payload=PromptAddedPayload(prompt_id=prompt_id, text=text, embed_ms=embed_ms),
            )
        )
        return prompt_id

    async def remove_prompt(self, prompt_id: str) -> None:
        async with self._prompts_lock:
            rec = self._prompts.pop(prompt_id, None)
        if rec is not None:
            await self._emit(
                PromptRemoved(
                    session_id=self.session_id,
                    source_id=None,
                    frame_id=None,
                    monotonic_ts=time.perf_counter(),
                    wall_ts=datetime.now(timezone.utc),
                    payload=PromptRemovedPayload(prompt_id=rec.prompt_id, text=rec.text),
                )
            )

    def list_prompts(self) -> list[dict]:
        return [{"prompt_id": r.prompt_id, "text": r.text} for r in self._prompts.values()]

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame) -> None:
        if self._closing:
            await self._emit_dropped(frame.frame_id, "session_closing")
            return

        # Backpressure: drop if more than one frame is already in flight
        # at the frame embedder (sustained backlog). Spectral analytics
        # is cheap; the GPU embedder is the bottleneck.
        if self._in_flight >= 2:
            self._frames_dropped += 1
            await self._emit_dropped(frame.frame_id, "embedder_busy")
            return

        self._in_flight += 1
        try:
            await self._process_frame_inner(frame)
        finally:
            self._in_flight -= 1

        self._frames_processed += 1

    async def _process_frame_inner(self, frame: Frame) -> None:
        t_frame_start = time.perf_counter()

        # 1) Frame embedding (if frame_embedder is configured)
        frame_emb: np.ndarray | None = None
        infer_ms = 0.0
        if self.frame_embedder is not None:
            t0 = time.perf_counter()
            try:
                arr = await self._call_embedder(
                    self.frame_embedder, [frame], role="frame"
                )
            except Exception as exc:
                await self._emit_embedder_error("frame", exc)
                return
            infer_ms = (time.perf_counter() - t0) * 1000.0
            frame_emb = arr[0]

        # 2) Spectral analytics
        # Emission order: ShotBoundary, AnomalyAlert first, then
        # FrameMetrics LAST as the per-frame flush signal so an
        # aggregating sink can render a single per-frame UI update.
        if frame_emb is not None and self._spectral is not None:
            update = self._spectral.update(frame_emb)
            if update.is_shot_boundary:
                await self._emit(
                    ShotBoundary(
                        session_id=self.session_id,
                        source_id=frame.source_id,
                        frame_id=frame.frame_id,
                        monotonic_ts=frame.monotonic_ts,
                        wall_ts=datetime.now(timezone.utc),
                        payload=ShotBoundaryPayload(
                            shot_count=update.shot_count,
                            anomaly_at_cut=update.anomaly_score,
                        ),
                    )
                )
            should_fire, sustained = self._anomaly_tracker.step(update.anomaly_score)
            if should_fire:
                await self._emit(
                    AnomalyAlert(
                        session_id=self.session_id,
                        source_id=frame.source_id,
                        frame_id=frame.frame_id,
                        monotonic_ts=frame.monotonic_ts,
                        wall_ts=datetime.now(timezone.utc),
                        payload=AnomalyAlertPayload(
                            anomaly_score=update.anomaly_score,
                            threshold=self._anomaly_tracker.threshold,
                            sustained_frames=sustained,
                        ),
                    )
                )
            await self._emit(
                FrameMetrics(
                    session_id=self.session_id,
                    source_id=frame.source_id,
                    frame_id=frame.frame_id,
                    monotonic_ts=frame.monotonic_ts,
                    wall_ts=datetime.now(timezone.utc),
                    payload=FrameMetricsPayload(
                        entropy_norm=update.entropy_norm,
                        motion_score=update.motion_score,
                        anomaly_score=update.anomaly_score,
                        buffer_fill=update.buffer_fill,
                        infer_ms=infer_ms,
                    ),
                )
            )

        # 3) Clip accumulation (fire-and-forget when full)
        if self.clip_embedder is not None and self._resolved_clip is not None:
            n_clip = self._resolved_clip.clip_frames
            n_stride = self._resolved_clip.clip_stride
            self._clip_accumulator.append(frame)
            if (
                len(self._clip_accumulator) >= n_clip
                and not self._clip_pending
            ):
                self._clip_pending = True
                clip_frames = self._clip_accumulator[:n_clip]
                # Advance by stride
                self._clip_accumulator = self._clip_accumulator[n_stride:]
                task = asyncio.create_task(self._embed_clip_and_emit(clip_frames))
                self._clip_tasks.add(task)
                task.add_done_callback(self._clip_tasks.discard)
            # Cap accumulator at 2x clip_frames to prevent unbounded growth
            cap = n_clip * 2
            if len(self._clip_accumulator) > cap:
                self._clip_accumulator = self._clip_accumulator[-n_clip:]

        # Frame processing wall-time (debug only — not emitted as event)
        _ = (time.perf_counter() - t_frame_start) * 1000.0

    async def _embed_clip_and_emit(self, frames: list[Frame]) -> None:
        try:
            t0 = time.perf_counter()
            assert self.clip_embedder is not None
            arr = await self._call_embedder(self.clip_embedder, [frames], role="clip")
            clip_emb = arr[0]
            clip_infer_ms = (time.perf_counter() - t0) * 1000.0

            # Score against all current prompts (snapshot to avoid races)
            async with self._prompts_lock:
                snapshot = list(self._prompts.values())
            scores: list[ClipScore] = []
            for rec in snapshot:
                sim = float(np.dot(clip_emb, rec.embedding))
                scores.append(
                    ClipScore(prompt_id=rec.prompt_id, text=rec.text, similarity=sim)
                )

            last_frame = frames[-1]
            await self._emit(
                ClipScores(
                    session_id=self.session_id,
                    source_id=last_frame.source_id,
                    frame_id=last_frame.frame_id,
                    monotonic_ts=last_frame.monotonic_ts,
                    wall_ts=datetime.now(timezone.utc),
                    payload=ClipScoresPayload(scores=scores, clip_infer_ms=clip_infer_ms),
                )
            )
        except Exception as exc:
            await self._emit_embedder_error("clip", exc)
        finally:
            self._clip_pending = False

    # ------------------------------------------------------------------
    # Embedder dispatch
    # ------------------------------------------------------------------

    async def _call_embedder(
        self,
        embedder: ImageEmbedder | VideoEmbedder | TextEmbedder,
        inputs: list,
        *,
        role: Literal["frame", "clip", "text"],
    ) -> np.ndarray:
        fn: Callable = embedder.embed_fn
        async with self._embed_sem:
            if inspect.iscoroutinefunction(fn):
                result: Any = await fn(inputs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self._embed_executor, fn, inputs)
                if inspect.isawaitable(result):
                    result = await result
        return _validate_and_normalize(
            result, expected_n=len(inputs), expected_dim=embedder.embed_dim, role=role
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit_dropped(self, frame_id: int, reason: str) -> None:
        await self._emit(
            FrameDropped(
                session_id=self.session_id,
                source_id=None,
                frame_id=frame_id,
                monotonic_ts=time.perf_counter(),
                wall_ts=datetime.now(timezone.utc),
                payload=FrameDroppedPayload(frame_id=frame_id, reason=reason),  # type: ignore[arg-type]
            )
        )

    async def _emit_embedder_error(self, role: str, exc: Exception) -> None:
        await self._emit(
            EmbedderError(
                session_id=self.session_id,
                source_id=None,
                frame_id=None,
                monotonic_ts=time.perf_counter(),
                wall_ts=datetime.now(timezone.utc),
                payload=EmbedderErrorPayload(
                    role=role,  # type: ignore[arg-type]
                    error_type=type(exc).__name__,
                    message=str(exc),
                ),
            )
        )

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped


# ---------------------------------------------------------------------------
# Embedder output validation
# ---------------------------------------------------------------------------


def _validate_and_normalize(
    result: Any,
    *,
    expected_n: int,
    expected_dim: int,
    role: str,
) -> np.ndarray:
    """Shape/finiteness check then L2-normalize. Raises EmbedderValidationError on bad."""
    arr = np.asarray(result)
    if arr.ndim != 2:
        raise EmbedderValidationError(
            role=role,
            expected=f"2-D ndarray (N, {expected_dim})",
            actual=f"ndim={arr.ndim}, shape={arr.shape}",
        )
    if arr.shape != (expected_n, expected_dim):
        raise EmbedderValidationError(
            role=role,
            expected=f"shape ({expected_n}, {expected_dim})",
            actual=f"shape {arr.shape}",
        )
    if not np.all(np.isfinite(arr)):
        raise EmbedderValidationError(
            role=role,
            expected="finite values",
            actual="NaN/Inf present",
        )
    return l2_normalize(arr.astype(np.float64))
