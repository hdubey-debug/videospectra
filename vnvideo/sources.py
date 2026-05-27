"""FrameSource protocol — the contract for anything that produces Frames.

v0.1 ships only the Protocol and a ``run_source`` convenience helper.
Concrete implementations (video file reader, webcam, RTSP) are deferred
to v0.1.1; the WebSocket frame ingestion path is implemented in
``vnvideo/server/routes.py``, not here, because it depends on FastAPI.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from vnvideo.session import Session
from vnvideo.types import Frame

logger = logging.getLogger(__name__)


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields Frames over time.

    Structurally typed: any object with the right attributes / methods
    satisfies the Protocol — no inheritance required.
    """

    source_id: str

    def frames(self) -> AsyncIterator[Frame]:
        ...

    async def aclose(self) -> None:
        ...


async def run_source(source: FrameSource, session: Session) -> None:
    """Pump frames from a source into a session until the source is done.

    Handles ``KeyboardInterrupt`` and ``asyncio.CancelledError`` by
    closing the source cleanly. The caller is responsible for closing
    the session.
    """
    try:
        async for frame in source.frames():
            await session.process_frame(frame)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("run_source interrupted; closing source %s", source.source_id)
        raise
    finally:
        try:
            await source.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("source %s aclose raised: %s", source.source_id, exc)
