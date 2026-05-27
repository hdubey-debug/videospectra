"""Core typed contracts: Frame, TaskType, Capability, Capabilities.

Pure Python, no I/O, no torch.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class TaskType(str, Enum):
    """High-level task hints that an embedder may use to pick an instruction.

    Strings (not opaque ints) so they serialize cleanly into events.
    """

    IMAGE_RETRIEVAL = "image_retrieval"
    VIDEO_RETRIEVAL = "video_retrieval"
    TEXT_SIMILARITY = "text_similarity"
    DOCUMENT_RETRIEVAL = "document_retrieval"


class Capability(str, Enum):
    """What an embedder or session can do.

    Used by ``Capabilities`` to advertise features to the UI / client.
    """

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    INSTRUCTION_AWARE = "instruction_aware"
    NORMALIZED_OUTPUT = "normalized_output"
    ASYNC_NATIVE = "async_native"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class Capabilities:
    """Session-level capability advertisement.

    The Session emits this once over the wire on connect so a client can
    decide which UI panels and inputs to enable.
    """

    spectral_analytics: bool
    clip_events: bool
    prompts: bool
    embedder_names: dict[str, str] = field(default_factory=dict)
    embedder_spaces: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """A single video frame plus identifying metadata.

    The frame is immutable. ``monotonic_ts`` (perf_counter) is used for
    latency math only; ``wall_ts`` (time.time) is used for logging and
    display only — they have different epochs and must not be mixed.

    No metadata escape hatch — passing context to an embedder is the
    caller's responsibility (e.g., via a closure in the setup file).
    """

    image: PILImage
    frame_id: int
    source_id: str
    monotonic_ts: float
    wall_ts: float
    width: int
    height: int

    @classmethod
    def from_jpeg(cls, data: bytes, *, source_id: str, frame_id: int) -> Frame:
        """Decode a JPEG byte string into a Frame, stamping current timestamps."""
        image = Image.open(io.BytesIO(data))
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return cls(
            image=image,
            frame_id=frame_id,
            source_id=source_id,
            monotonic_ts=time.perf_counter(),
            wall_ts=time.time(),
            width=image.width,
            height=image.height,
        )

    @classmethod
    def from_pil(cls, image: PILImage, *, source_id: str, frame_id: int) -> Frame:
        """Wrap an existing PIL image, stamping current timestamps.

        The image is converted to RGB if necessary. The result is a new
        Frame whose ``image`` attribute may be the same object the caller
        passed in (if already RGB) or a converted copy.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return cls(
            image=image,
            frame_id=frame_id,
            source_id=source_id,
            monotonic_ts=time.perf_counter(),
            wall_ts=time.time(),
            width=image.width,
            height=image.height,
        )
