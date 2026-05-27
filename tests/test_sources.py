"""Tests for vnvideo.sources — Protocol shape + run_source helper."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from PIL import Image

from vnvideo import sources as sources_mod
from vnvideo.embedders import DummyEmbedder
from vnvideo.session import Session
from vnvideo.sources import FrameSource, run_source
from vnvideo.types import Frame


class _InlineSource:
    source_id = "test"

    def __init__(self, n: int) -> None:
        self.n = n
        self.closed = False

    async def frames(self) -> AsyncIterator[Frame]:
        for i in range(self.n):
            yield Frame.from_pil(
                Image.new("RGB", (4, 4), (i, i, i)),
                source_id=self.source_id,
                frame_id=i,
            )

    async def aclose(self) -> None:
        self.closed = True


class TestProtocolShape:
    def test_inline_class_satisfies_protocol(self) -> None:
        src = _InlineSource(n=3)
        assert isinstance(src, FrameSource)

    def test_module_does_not_import_fastapi(self) -> None:
        with open(sources_mod.__file__) as f:
            source = f.read()
        assert "import fastapi" not in source
        assert "from fastapi" not in source


class TestRunSource:
    async def test_pumps_all_frames(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        await run_source(_InlineSource(n=5), session)
        assert session.frames_processed == 5
        await session.aclose()

    async def test_closes_source_on_exit(self) -> None:
        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        src = _InlineSource(n=3)
        await run_source(src, session)
        assert src.closed is True
        await session.aclose()

    async def test_closes_source_on_cancellation(self) -> None:
        # A source that never yields and waits indefinitely
        cancelled_close = asyncio.Event()

        class _HangingSource:
            source_id = "hang"

            async def frames(self) -> AsyncIterator[Frame]:
                await asyncio.sleep(100)
                if False:
                    yield  # type: ignore[unreachable]

            async def aclose(self) -> None:
                cancelled_close.set()

        session = Session(frame_embedder=DummyEmbedder.make_image())
        await session.start()
        task = asyncio.create_task(run_source(_HangingSource(), session))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled_close.is_set()
        await session.aclose()
