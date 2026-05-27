"""Tests for vnvideo.session — orchestrator, validation, embedder dispatch."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
from PIL import Image

from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import DummyEmbedder, ImageEmbedder, TextEmbedder, VideoEmbedder
from vnvideo.events import (
    ClipScores,
    EmbedderError,
    Event,
    FrameDropped,
    FrameMetrics,
    PromptAdded,
    SessionInfo,
)
from vnvideo.session import (
    ClipConfig,
    EmbedderValidationError,
    Session,
)
from vnvideo.sinks import MemorySink
from vnvideo.types import Frame


def _make_frame(i: int, color: tuple[int, int, int] = (128, 64, 32)) -> Frame:
    return Frame.from_pil(Image.new("RGB", (16, 16), color), source_id="t", frame_id=i)


async def _drain(sink: MemorySink) -> list[Event]:
    """Pull all events currently in the memory sink (after aclose)."""
    out: list[Event] = []
    while True:
        try:
            ev = await sink.get(timeout=0.05)
        except (StopAsyncIteration, asyncio.TimeoutError):
            return out
        out.append(ev)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_at_least_one_embedder(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Session()

    def test_clip_without_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="text_embedder"):
            Session(clip_embedder=DummyEmbedder.make_video(), clip_config=ClipConfig())

    def test_text_without_clip_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip_embedder"):
            Session(text_embedder=DummyEmbedder.make_text())

    def test_clip_text_dim_mismatch_rejected(self) -> None:
        # Create a text embedder with a different dim
        bad_text = TextEmbedder(
            embed_dim=64,
            space_id="vnvideo/dummy@128/test",  # same space but wrong dim
            embed_fn=lambda t: np.zeros((len(t), 64)),
        )
        with pytest.raises(ValueError, match="embed_dim"):
            Session(
                clip_embedder=DummyEmbedder.make_video(),
                text_embedder=bad_text,
                clip_config=ClipConfig(),
            )

    def test_clip_text_space_mismatch_rejected(self) -> None:
        bad_text = TextEmbedder(
            embed_dim=128,
            space_id="other/space@128/text",
            embed_fn=lambda t: np.zeros((len(t), 128)),
        )
        with pytest.raises(ValueError, match="space_id"):
            Session(
                clip_embedder=DummyEmbedder.make_video(),
                text_embedder=bad_text,
                clip_config=ClipConfig(),
            )

    def test_clip_without_clip_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip_config"):
            Session(
                clip_embedder=DummyEmbedder.make_video(),
                text_embedder=DummyEmbedder.make_text(),
            )


# ---------------------------------------------------------------------------
# Lifecycle & capabilities
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_emits_session_info(self) -> None:
        sink = MemorySink()
        session = Session(frame_embedder=DummyEmbedder.make_image(), sinks=[sink])
        await session.start()
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        assert any(isinstance(e, SessionInfo) for e in events)

    async def test_capabilities_reflects_config(self) -> None:
        session = Session(
            frame_embedder=DummyEmbedder.make_image(),
            clip_embedder=DummyEmbedder.make_video(),
            text_embedder=DummyEmbedder.make_text(),
            spectral_config=SpectralConfig(),
            clip_config=ClipConfig(),
        )
        caps = session.capabilities()
        assert caps.spectral_analytics is True
        assert caps.clip_events is True
        assert caps.prompts is True
        assert "frame" in caps.embedder_names
        assert "clip" in caps.embedder_names
        assert "text" in caps.embedder_names

    async def test_capabilities_spectral_off_without_config(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        caps = session.capabilities()
        # No spectral_config -> spectral disabled
        assert caps.spectral_analytics is False
        assert caps.clip_events is False
        assert caps.prompts is False


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    async def test_warmup_succeeds_with_good_embedder(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        await session.warmup()
        await session.aclose()

    async def test_warmup_fails_on_bad_dim(self) -> None:
        bad_embedder = ImageEmbedder(
            embed_dim=128,
            space_id="bad/embedder@128/test",
            embed_fn=lambda frames: np.zeros((len(frames), 64)),  # wrong dim
        )
        session = Session(frame_embedder=bad_embedder)
        await session.start()
        with pytest.raises(EmbedderValidationError, match="shape"):
            await session.warmup()
        await session.aclose()

    async def test_warmup_fails_on_nan(self) -> None:
        nan_embedder = ImageEmbedder(
            embed_dim=128,
            space_id="nan/embedder@128/test",
            embed_fn=lambda frames: np.full((len(frames), 128), np.nan),
        )
        session = Session(frame_embedder=nan_embedder)
        await session.start()
        with pytest.raises(EmbedderValidationError, match="NaN"):
            await session.warmup()
        await session.aclose()


# ---------------------------------------------------------------------------
# Frame loop
# ---------------------------------------------------------------------------


class TestFrameLoop:
    async def test_processes_frame_and_emits_metrics(self) -> None:
        sink = MemorySink()
        session = Session(
            frame_embedder=DummyEmbedder.make_image(),
            sinks=[sink],
            spectral_config=SpectralConfig(window_frames=5),
        )
        await session.start()
        await session.warmup()
        for i in range(10):
            await session.process_frame(_make_frame(i))
        await asyncio.sleep(0.05)
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        metric_events = [e for e in events if isinstance(e, FrameMetrics)]
        assert len(metric_events) == 10

    async def test_spectral_off_when_no_config(self) -> None:
        sink = MemorySink()
        session = Session(frame_embedder=DummyEmbedder.make_image(), sinks=[sink])
        await session.start()
        for i in range(5):
            await session.process_frame(_make_frame(i))
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        # No spectral config -> no FrameMetrics events
        metric_events = [e for e in events if isinstance(e, FrameMetrics)]
        assert metric_events == []

    async def test_frames_processed_counter(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        for i in range(7):
            await session.process_frame(_make_frame(i))
        assert session.frames_processed == 7
        await session.aclose()


# ---------------------------------------------------------------------------
# Embedder concurrency
# ---------------------------------------------------------------------------


class TestEmbedderConcurrency:
    async def test_concurrency_1_serializes_calls(self) -> None:
        """With embedder_concurrency=1, two embed calls must not overlap."""
        in_flight = 0
        max_seen = 0

        def slow_embed_fn(frames: list[Frame]) -> np.ndarray:
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            time.sleep(0.05)
            in_flight -= 1
            return np.random.randn(len(frames), 64).astype(np.float64)

        emb = ImageEmbedder(
            embed_dim=64,
            space_id="test/slow@64/test",
            embed_fn=slow_embed_fn,
        )
        session = Session(frame_embedder=emb, embedder_concurrency=1)
        await session.start()
        # Two parallel calls; should serialize via semaphore
        await asyncio.gather(
            session.process_frame(_make_frame(0)),
            session.process_frame(_make_frame(1)),
        )
        await session.aclose()
        assert max_seen == 1


# ---------------------------------------------------------------------------
# Fire-and-forget clip embedding
# ---------------------------------------------------------------------------


class TestClipEmbedding:
    async def test_clip_fires_when_accumulator_full(self) -> None:
        sink = MemorySink()
        session = Session(
            frame_embedder=DummyEmbedder.make_image(),
            clip_embedder=DummyEmbedder.make_video(),
            text_embedder=DummyEmbedder.make_text(),
            sinks=[sink],
            clip_config=ClipConfig(clip_frames=4, clip_stride=4),
        )
        await session.start()
        for i in range(8):
            await session.process_frame(_make_frame(i))
        # Wait for clip tasks to finish
        await asyncio.sleep(0.2)
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        clip_events = [e for e in events if isinstance(e, ClipScores)]
        # 8 frames / 4 per clip = 2 clips
        assert len(clip_events) >= 1

    async def test_slow_clip_does_not_block_frame_loop(self) -> None:
        """A clip_embed that takes 200ms must not stall frame processing."""

        clip_started = asyncio.Event()
        clip_release = asyncio.Event()

        async def slow_clip_embed(clips: list[list[Frame]]) -> np.ndarray:
            clip_started.set()
            await clip_release.wait()
            return np.random.randn(len(clips), 128).astype(np.float64)

        slow_clip = VideoEmbedder(
            embed_dim=128,
            space_id="vnvideo/dummy@128/test",
            embed_fn=slow_clip_embed,
        )

        sink = MemorySink()
        session = Session(
            frame_embedder=DummyEmbedder.make_image(),
            clip_embedder=slow_clip,
            text_embedder=DummyEmbedder.make_text(),
            sinks=[sink],
            clip_config=ClipConfig(clip_frames=4, clip_stride=4),
            embedder_concurrency=2,  # Don't serialize frame and clip
        )
        await session.start()

        # Push enough frames to start a clip embed task
        for i in range(4):
            await session.process_frame(_make_frame(i))
        # Wait until the slow clip embed has started
        await asyncio.wait_for(clip_started.wait(), timeout=2.0)

        # Now push more frames — these should still process quickly
        # because clip embed runs on a fire-and-forget task
        t0 = time.perf_counter()
        for i in range(4, 8):
            await session.process_frame(_make_frame(i))
        elapsed = time.perf_counter() - t0

        # 4 fast frames should be much less than the 200ms+ clip embed
        # (which is still pending). Allow generous slack for CI noise.
        assert elapsed < 1.0, f"frame loop blocked by clip embed: {elapsed:.3f}s"

        clip_release.set()
        await asyncio.sleep(0.2)
        await session.aclose()
        await sink.aclose()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    async def test_add_prompt_emits_event(self) -> None:
        sink = MemorySink()
        session = Session(
            clip_embedder=DummyEmbedder.make_video(),
            text_embedder=DummyEmbedder.make_text(),
            sinks=[sink],
            clip_config=ClipConfig(),
        )
        await session.start()
        pid = await session.add_prompt("a cat playing")
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        prompt_events = [e for e in events if isinstance(e, PromptAdded)]
        assert len(prompt_events) == 1
        assert prompt_events[0].payload.prompt_id == pid

    async def test_list_prompts(self) -> None:
        session = Session(
            clip_embedder=DummyEmbedder.make_video(),
            text_embedder=DummyEmbedder.make_text(),
            clip_config=ClipConfig(),
        )
        await session.start()
        p1 = await session.add_prompt("dog")
        p2 = await session.add_prompt("cat")
        prompts = session.list_prompts()
        ids = {p["prompt_id"] for p in prompts}
        assert ids == {p1, p2}
        await session.aclose()

    async def test_remove_prompt(self) -> None:
        session = Session(
            clip_embedder=DummyEmbedder.make_video(),
            text_embedder=DummyEmbedder.make_text(),
            clip_config=ClipConfig(),
        )
        await session.start()
        pid = await session.add_prompt("dog")
        await session.remove_prompt(pid)
        assert session.list_prompts() == []
        await session.aclose()

    async def test_add_prompt_without_text_embedder_raises(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        with pytest.raises(RuntimeError, match="no text_embedder"):
            await session.add_prompt("x")
        await session.aclose()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_failing_embedder_emits_error_event(self) -> None:
        def raising(frames: list[Frame]) -> np.ndarray:
            raise RuntimeError("intentional failure")

        bad_emb = ImageEmbedder(
            embed_dim=64,
            space_id="bad/raise@64/test",
            embed_fn=raising,
        )
        sink = MemorySink()
        session = Session(frame_embedder=bad_emb, sinks=[sink])
        await session.start()
        await session.process_frame(_make_frame(0))
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        errs = [e for e in events if isinstance(e, EmbedderError)]
        assert len(errs) >= 1
        assert errs[0].payload.role == "frame"
        assert "intentional failure" in errs[0].payload.message


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    async def test_frames_drop_when_in_flight_too_high(self) -> None:
        # Manually drive backpressure by pretending two frames are already
        # in flight when a third arrives.
        sink = MemorySink()
        session = Session(frame_embedder=DummyEmbedder.make_image(), sinks=[sink])
        await session.start()
        session._in_flight = 2  # type: ignore[attr-defined]
        await session.process_frame(_make_frame(99))
        session._in_flight = 0  # type: ignore[attr-defined]
        await session.aclose()
        await sink.aclose()
        events = await _drain(sink)
        dropped = [e for e in events if isinstance(e, FrameDropped)]
        assert len(dropped) == 1
        assert dropped[0].payload.reason == "embedder_busy"


# ---------------------------------------------------------------------------
# Async embedder dispatch
# ---------------------------------------------------------------------------


class TestAsyncEmbedder:
    async def test_async_embed_fn_awaited(self) -> None:
        async def async_embed(frames: list[Frame]) -> np.ndarray:
            await asyncio.sleep(0.01)
            return np.random.randn(len(frames), 64).astype(np.float64)

        emb = ImageEmbedder(embed_dim=64, space_id="async/x@64/t", embed_fn=async_embed)
        session = Session(frame_embedder=emb)
        await session.start()
        await session.process_frame(_make_frame(0))
        assert session.frames_processed == 1
        await session.aclose()
