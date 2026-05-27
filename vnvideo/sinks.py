"""Event distribution layer.

A ``Sink`` is anything that consumes events. The Session never calls
``sink.emit`` directly — every registered sink is wrapped in a
``SinkRunner`` with a bounded queue and a per-sink worker task, so a
slow or failing sink can never stall the frame loop.

Built-in sinks:

- ``StdoutSink`` — one JSON line per event
- ``JsonlSink(path)`` — append-only NDJSON file
- ``MemorySink`` — ``asyncio.Queue``-backed; iterable; the canonical
  SDK / test consumption pattern
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, Literal, Protocol, runtime_checkable

from vnvideo.events import Event

logger = logging.getLogger(__name__)

OverflowPolicy = Literal["drop_oldest", "drop_newest", "block"]


@runtime_checkable
class Sink(Protocol):
    """A consumer of events.

    Implementations are arbitrary — emit may post to a webhook, write
    to a DB, render to a UI, push to a websocket, etc. The framework
    only cares that ``emit`` is awaitable and ``aclose`` cleans up.
    """

    name: str

    async def emit(self, event: Event) -> None:
        ...

    async def aclose(self) -> None:
        ...


# ---------------------------------------------------------------------------
# SinkRunner — bounded queue + per-sink worker + failure isolation
# ---------------------------------------------------------------------------


class SinkRunner:
    """Wraps a ``Sink`` with a bounded queue and a worker task.

    Enqueue is the only operation the Session performs; the worker
    drains the queue independently and calls ``sink.emit`` one event
    at a time, so a slow sink only causes drops (not stalls) and a
    consistently-failing sink is disabled instead of crashing the
    session.

    Overflow policy:

    - ``drop_oldest`` (default): evict oldest queued event to make
      room. Best for live dashboards where latest matters more.
    - ``drop_newest``: discard the new event. Best for first-write-
      wins recording.
    - ``block``: ``enqueue`` awaits room in the queue. Best for sinks
      that must not lose events (e.g., JSONL recording in tests).

    A sink that raises ``failure_threshold`` consecutive times is
    marked disabled. New events skip enqueue entirely until shutdown.
    """

    def __init__(
        self,
        sink: Sink,
        *,
        max_queue: int = 100,
        overflow: OverflowPolicy = "drop_oldest",
        failure_threshold: int = 5,
        drain_timeout: float = 5.0,
    ) -> None:
        if max_queue <= 0:
            raise ValueError("max_queue must be a positive integer")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be a positive integer")

        self.sink = sink
        self.max_queue = max_queue
        self.overflow: OverflowPolicy = overflow
        self.failure_threshold = failure_threshold
        self.drain_timeout = drain_timeout

        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._disabled = False
        self._failure_count = 0
        self._closing = False
        self._dropped_count = 0

    @property
    def name(self) -> str:
        return self.sink.name

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def dropped_count(self) -> int:
        """Total events dropped at this runner due to overflow."""
        return self._dropped_count

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._worker(), name=f"sink-{self.name}")

    async def enqueue(self, event: Event) -> None:
        if self._disabled or self._closing:
            return

        if self.overflow == "block":
            await self._queue.put(event)
            return

        # drop policies: non-blocking
        try:
            self._queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        if self.overflow == "drop_oldest":
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Lost the race against a concurrent put; count as drop.
                self._dropped_count += 1
        else:  # drop_newest
            self._dropped_count += 1

    async def _worker(self) -> None:
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self.sink.emit(event)
                self._failure_count = 0
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 — isolate sink failures by design
                self._failure_count += 1
                logger.warning(
                    "sink %s raised (%d/%d): %s",
                    self.name,
                    self._failure_count,
                    self.failure_threshold,
                    exc,
                )
                if self._failure_count >= self.failure_threshold:
                    self._disabled = True
                    logger.error(
                        "sink %s disabled after %d consecutive failures",
                        self.name,
                        self._failure_count,
                    )
                    return
            finally:
                self._queue.task_done()

    async def aclose(self) -> None:
        self._closing = True
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._queue.join(), timeout=self.drain_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "sink %s did not drain in %.1fs; %d events remain",
                    self.name,
                    self.drain_timeout,
                    self._queue.qsize(),
                )
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        try:
            await self.sink.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("sink %s aclose raised: %s", self.name, exc)


# ---------------------------------------------------------------------------
# Built-in sinks
# ---------------------------------------------------------------------------


class StdoutSink:
    """Write one JSON line per event to stdout (or a custom stream).

    Useful for piping into ``jq`` or for container logs.
    """

    name: str = "stdout"

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream: IO[str] = stream if stream is not None else sys.stdout

    async def emit(self, event: Event) -> None:
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()

    async def aclose(self) -> None:
        pass


class JsonlSink:
    """Append-only newline-delimited JSON file sink.

    File writes happen on a thread executor so the event loop is not
    blocked. The file is opened lazily on first emit and re-opened in
    append mode across process restarts.
    """

    name: str = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fp: IO[str] | None = None

    def _open(self) -> IO[str]:
        if self._fp is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.path.open("a", encoding="utf-8")
        return self._fp

    async def emit(self, event: Event) -> None:
        line = event.model_dump_json() + "\n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_line, line)

    def _write_line(self, line: str) -> None:
        fp = self._open()
        fp.write(line)
        fp.flush()

    async def aclose(self) -> None:
        if self._fp is not None:
            loop = asyncio.get_running_loop()
            fp = self._fp
            self._fp = None
            await loop.run_in_executor(None, fp.close)


class MemorySink:
    """In-memory queue sink. The canonical way to consume events in
    tests and SDK use cases.

    Iterate with ``async for event in sink:`` to receive events as
    they arrive. After ``aclose()`` the iterator stops cleanly.
    """

    name: str = "memory"

    def __init__(self, maxsize: int = 0) -> None:
        # maxsize=0 means unbounded; vanilla asyncio.Queue convention
        self._queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._sentinel = object()

    async def emit(self, event: Event) -> None:
        if self._closed:
            return
        await self._queue.put(event)

    async def get(self, timeout: float | None = None) -> Event:
        """Synchronously (well, awaitably) pull one event.

        Returns the next event or raises ``asyncio.TimeoutError`` when
        ``timeout`` is set and elapses, or raises ``StopAsyncIteration``
        if the sink has been closed and drained.
        """
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is self._sentinel:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        item = await self._queue.get()
        if item is self._sentinel:
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(self._sentinel)

    def qsize(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def event_to_json(event: Event) -> str:
    """Stable JSON serialization with sorted keys (for snapshot tests)."""
    return json.dumps(event.model_dump(mode="json"), sort_keys=True)
