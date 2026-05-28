"""Tests for videospectra.sinks — Sink protocol, SinkRunner backpressure, built-ins."""
from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from videospectra.events import (
    Event,
    FrameMetrics,
    FrameMetricsPayload,
    Status,
    StatusPayload,
)
from videospectra.sinks import (
    JsonlSink,
    MemorySink,
    Sink,
    SinkRunner,
    StdoutSink,
)


def _make_status(message: str = "hi", frame_id: int = 0) -> Event:
    return Status(
        session_id="test",
        source_id="t",
        frame_id=frame_id,
        monotonic_ts=0.0,
        wall_ts=datetime(2026, 5, 27, tzinfo=timezone.utc),
        payload=StatusPayload(message=message),
    )


def _make_metric(frame_id: int) -> Event:
    return FrameMetrics(
        session_id="test",
        source_id="t",
        frame_id=frame_id,
        monotonic_ts=float(frame_id),
        wall_ts=datetime(2026, 5, 27, tzinfo=timezone.utc),
        payload=FrameMetricsPayload(
            entropy_norm=0.1,
            motion_score=0.1,
            anomaly_score=0.1,
            buffer_fill=1,
            infer_ms=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Test sinks (collect / slow / fail)
# ---------------------------------------------------------------------------


class _Collector:
    name = "collector"

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        pass


class _SlowSink:
    name = "slow"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        await asyncio.sleep(self.delay)
        self.events.append(event)

    async def aclose(self) -> None:
        pass


class _AlwaysFailSink:
    name = "fail"

    def __init__(self) -> None:
        self.attempts = 0

    async def emit(self, event: Event) -> None:
        self.attempts += 1
        raise RuntimeError(f"intentional failure #{self.attempts}")

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


class TestSinkProtocol:
    def test_collector_satisfies_protocol(self) -> None:
        assert isinstance(_Collector(), Sink)

    def test_stdout_sink_satisfies_protocol(self) -> None:
        assert isinstance(StdoutSink(), Sink)


# ---------------------------------------------------------------------------
# SinkRunner — backpressure & policies
# ---------------------------------------------------------------------------


class TestSinkRunnerBasic:
    async def test_basic_emit(self) -> None:
        sink = _Collector()
        runner = SinkRunner(sink)
        runner.start()
        await runner.enqueue(_make_status("a"))
        await runner.enqueue(_make_status("b"))
        await asyncio.sleep(0.05)
        await runner.aclose()
        assert len(sink.events) == 2

    async def test_validation_rejects_zero_max_queue(self) -> None:
        with pytest.raises(ValueError):
            SinkRunner(_Collector(), max_queue=0)

    async def test_validation_rejects_zero_failure_threshold(self) -> None:
        with pytest.raises(ValueError):
            SinkRunner(_Collector(), failure_threshold=0)


class TestSinkRunnerOverflow:
    async def test_drop_oldest_keeps_latest(self) -> None:
        # Slow sink + small queue + many enqueues -> only the latest survive.
        sink = _SlowSink(delay=1.0)
        runner = SinkRunner(sink, max_queue=3, overflow="drop_oldest", drain_timeout=2.0)
        runner.start()
        # Enqueue 10 metrics. Worker is stuck for 1s on first emit; the
        # other 9 pile up into a queue of size 3 with drop_oldest, so
        # frame_ids 7,8,9 should be the ones still queued.
        for i in range(10):
            await runner.enqueue(_make_metric(i))
        # Wait long enough for at most a couple emits
        await asyncio.sleep(0.05)
        # The first metric is in flight; remaining queue has at most 3 items.
        assert runner._queue.qsize() <= 3
        assert runner.dropped_count >= 6
        await runner.aclose()

    async def test_drop_newest_keeps_oldest(self) -> None:
        sink = _SlowSink(delay=1.0)
        runner = SinkRunner(sink, max_queue=3, overflow="drop_newest")
        runner.start()
        for i in range(10):
            await runner.enqueue(_make_metric(i))
        await asyncio.sleep(0.05)
        # The first metric is in flight; first 3 of the remaining are kept.
        # frame_ids that survive (queued or in-flight): {0, 1, 2, 3}
        assert runner.dropped_count >= 6
        await runner.aclose()

    async def test_block_policy_serializes(self) -> None:
        sink = _SlowSink(delay=0.05)
        runner = SinkRunner(sink, max_queue=2, overflow="block")
        runner.start()
        for i in range(6):
            await runner.enqueue(_make_metric(i))
        # Wait for worker to drain
        await asyncio.sleep(0.5)
        await runner.aclose()
        assert len(sink.events) == 6
        assert runner.dropped_count == 0


class TestSinkRunnerFailures:
    async def test_failing_sink_disabled_after_threshold(self) -> None:
        sink = _AlwaysFailSink()
        runner = SinkRunner(sink, max_queue=10, failure_threshold=3)
        runner.start()
        for i in range(10):
            await runner.enqueue(_make_metric(i))
        await asyncio.sleep(0.1)
        assert runner.is_disabled
        # Exactly 3 attempts (= threshold) — worker stops after disabling
        assert sink.attempts == 3
        # Further enqueues are no-ops
        before = runner._queue.qsize()
        await runner.enqueue(_make_metric(100))
        assert runner._queue.qsize() == before
        await runner.aclose()

    async def test_transient_failures_reset_counter(self) -> None:
        class _FlakySink:
            name = "flaky"

            def __init__(self) -> None:
                self.calls = 0

            async def emit(self, event: Event) -> None:
                self.calls += 1
                if self.calls in (1, 3):
                    raise RuntimeError("flake")

            async def aclose(self) -> None:
                pass

        sink = _FlakySink()
        runner = SinkRunner(sink, max_queue=10, failure_threshold=3)
        runner.start()
        for i in range(5):
            await runner.enqueue(_make_metric(i))
        await asyncio.sleep(0.1)
        await runner.aclose()
        # Sink had non-consecutive failures, so it should NOT be disabled
        assert not runner.is_disabled


class TestSinkRunnerLifecycle:
    async def test_aclose_drains_pending(self) -> None:
        sink = _Collector()
        runner = SinkRunner(sink, max_queue=100)
        runner.start()
        for i in range(20):
            await runner.enqueue(_make_metric(i))
        await runner.aclose()
        # Drain should let all 20 emit
        assert len(sink.events) == 20

    async def test_aclose_timeout_on_stuck_sink(self) -> None:
        sink = _SlowSink(delay=10.0)
        runner = SinkRunner(sink, max_queue=5, drain_timeout=0.1)
        runner.start()
        for i in range(5):
            await runner.enqueue(_make_metric(i))
        # Should cancel and return within ~0.1s rather than hanging
        await asyncio.wait_for(runner.aclose(), timeout=1.0)


# ---------------------------------------------------------------------------
# StdoutSink
# ---------------------------------------------------------------------------


class TestStdoutSink:
    async def test_writes_jsonl(self) -> None:
        buf = io.StringIO()
        sink = StdoutSink(stream=buf)
        await sink.emit(_make_status("hello"))
        await sink.emit(_make_status("world"))
        await sink.aclose()
        lines = [line for line in buf.getvalue().splitlines() if line]
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["payload"]["message"] == "hello"
        assert parsed[1]["payload"]["message"] == "world"


# ---------------------------------------------------------------------------
# JsonlSink
# ---------------------------------------------------------------------------


class TestJsonlSink:
    async def test_appends_to_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        sink = JsonlSink(path)
        await sink.emit(_make_status("a"))
        await sink.emit(_make_status("b"))
        await sink.aclose()
        lines = path.read_text().splitlines()
        assert len(lines) == 2

    async def test_append_across_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        sink1 = JsonlSink(path)
        await sink1.emit(_make_status("a"))
        await sink1.aclose()
        sink2 = JsonlSink(path)
        await sink2.emit(_make_status("b"))
        await sink2.aclose()
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["payload"]["message"] == "a"
        assert json.loads(lines[1])["payload"]["message"] == "b"

    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "events.jsonl"
        sink = JsonlSink(path)
        await sink.emit(_make_status("hi"))
        await sink.aclose()
        assert path.exists()


# ---------------------------------------------------------------------------
# MemorySink
# ---------------------------------------------------------------------------


class TestMemorySink:
    async def test_emit_and_get(self) -> None:
        sink = MemorySink()
        await sink.emit(_make_status("a"))
        ev = await sink.get(timeout=0.5)
        assert isinstance(ev, Status)
        assert ev.payload.message == "a"

    async def test_async_iteration(self) -> None:
        sink = MemorySink()
        await sink.emit(_make_status("one"))
        await sink.emit(_make_status("two"))
        await sink.aclose()
        collected = [ev async for ev in sink]
        assert [ev.payload.message for ev in collected] == ["one", "two"]

    async def test_aclose_signals_stop(self) -> None:
        sink = MemorySink()
        await sink.aclose()
        with pytest.raises(StopAsyncIteration):
            await sink.get()


# ---------------------------------------------------------------------------
# End-to-end: SinkRunner + Memory consumer
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_runner_to_memory_sink(self) -> None:
        # Drives a memory sink directly through a SinkRunner: every
        # enqueued event must be observable in the same order via the
        # async iterator.
        sink = MemorySink()
        runner = SinkRunner(sink, max_queue=10, overflow="block")
        runner.start()
        for i in range(5):
            await runner.enqueue(_make_metric(i))
        await asyncio.sleep(0.05)
        await runner.aclose()
        await sink.aclose()
        seen: list[int] = []
        async for ev in sink:
            assert isinstance(ev, FrameMetrics)
            assert ev.frame_id is not None
            seen.append(ev.frame_id)
        assert seen == [0, 1, 2, 3, 4]
