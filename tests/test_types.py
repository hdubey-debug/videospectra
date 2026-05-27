"""Tests for vnvideo.types — Frame, Capabilities, enums."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from vnvideo.types import Capabilities, Capability, Frame, TaskType


def _make_jpeg_bytes(width: int = 64, height: int = 48, color: tuple[int, int, int] = (128, 64, 32)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class TestFrame:
    def test_frame_is_frozen(self) -> None:
        f = Frame.from_pil(Image.new("RGB", (4, 4)), source_id="s", frame_id=0)
        with pytest.raises(Exception):
            f.frame_id = 1  # type: ignore[misc]

    def test_frame_image_field_is_not_silently_swapped(self) -> None:
        original = Image.new("RGB", (4, 4))
        f = Frame.from_pil(original, source_id="s", frame_id=0)
        with pytest.raises(Exception):
            f.image = Image.new("RGB", (8, 8))  # type: ignore[misc]

    def test_from_jpeg_roundtrip(self) -> None:
        data = _make_jpeg_bytes(width=32, height=16)
        f = Frame.from_jpeg(data, source_id="cam0", frame_id=7)
        assert f.frame_id == 7
        assert f.source_id == "cam0"
        assert f.width == 32
        assert f.height == 16
        assert f.image.mode == "RGB"

    def test_from_pil_dimensions(self) -> None:
        img = Image.new("RGB", (100, 200))
        f = Frame.from_pil(img, source_id="cam1", frame_id=3)
        assert f.width == 100
        assert f.height == 200
        assert f.frame_id == 3

    def test_from_pil_converts_non_rgb(self) -> None:
        gray = Image.new("L", (10, 10))
        f = Frame.from_pil(gray, source_id="cam", frame_id=0)
        assert f.image.mode == "RGB"

    def test_timestamps_distinct_epochs(self) -> None:
        f = Frame.from_pil(Image.new("RGB", (4, 4)), source_id="s", frame_id=0)
        # monotonic_ts is from perf_counter (small offset from boot),
        # wall_ts is from time.time (Unix epoch, > 1e9 in any year >= 2001).
        assert f.wall_ts > 1e9
        assert f.monotonic_ts != f.wall_ts


class TestCapabilities:
    def test_default_empty(self) -> None:
        c = Capabilities(spectral_analytics=False, clip_events=False, prompts=False)
        assert c.embedder_names == {}
        assert c.embedder_spaces == {}

    def test_equality(self) -> None:
        c1 = Capabilities(
            spectral_analytics=True,
            clip_events=True,
            prompts=True,
            embedder_names={"frame": "rzen"},
            embedder_spaces={"frame": "qihoo/rzen@3584/video"},
        )
        c2 = Capabilities(
            spectral_analytics=True,
            clip_events=True,
            prompts=True,
            embedder_names={"frame": "rzen"},
            embedder_spaces={"frame": "qihoo/rzen@3584/video"},
        )
        assert c1 == c2

    def test_frozen(self) -> None:
        c = Capabilities(spectral_analytics=False, clip_events=False, prompts=False)
        with pytest.raises(Exception):
            c.spectral_analytics = True  # type: ignore[misc]


class TestEnums:
    def test_tasktype_values_are_strings(self) -> None:
        assert TaskType.IMAGE_RETRIEVAL.value == "image_retrieval"
        assert TaskType.VIDEO_RETRIEVAL.value == "video_retrieval"

    def test_capability_values_are_strings(self) -> None:
        assert Capability.IMAGE.value == "image"
        assert Capability.ASYNC_NATIVE.value == "async_native"
