"""BYOM (Bring Your Own Model) embedder wrappers and built-in demos.

Three dataclass wrappers — ``ImageEmbedder``, ``VideoEmbedder``,
``TextEmbedder`` — bundle a typed callable with the metadata the
framework needs to validate output shapes and check embedding-space
compatibility. The framework normalizes every embedding at the Session
boundary; the wrapper has no ``normalized`` flag.

Two built-in factories:

- ``DummyEmbedder.make_*`` — deterministic random-projection vectors,
  for tests. No semantic meaning.
- ``ColorHistogramEmbedder.make_image`` — real per-channel + grayscale
  histograms, 256-d, for the ``videospectra demo`` install-verification path.
  Pure numpy, no torch.
"""
from __future__ import annotations

import hashlib
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from videospectra import spaces
from videospectra.types import Frame

# A BYOM callable may return an ndarray directly (sync) or a coroutine
# resolving to one (async). ``inspect.isawaitable`` + ``run_in_executor``
# handle both at dispatch time.
MaybeAwaitable = Union[np.ndarray, Awaitable[np.ndarray]]

ImageEmbedFn = Callable[[list[Frame]], MaybeAwaitable]
VideoEmbedFn = Callable[[list[list[Frame]]], MaybeAwaitable]
TextEmbedFn = Callable[[list[str]], MaybeAwaitable]


@dataclass
class ImageEmbedder:
    """Per-frame image embedder.

    ``embed_fn`` receives ``list[Frame]`` (one per frame in the batch)
    and returns ``np.ndarray`` of shape ``(N, embed_dim)``. May be sync
    or async.
    """

    embed_dim: int
    space_id: str
    embed_fn: ImageEmbedFn
    name: str = "user"

    def __post_init__(self) -> None:
        _validate_common(self.embed_dim, self.space_id, self.name)


@dataclass
class VideoEmbedder:
    """Clip-level video embedder.

    ``embed_fn`` receives ``list[list[Frame]]`` — the outer list is a
    batch of clips, the inner list is the frames of one clip — and
    returns ``np.ndarray`` of shape ``(N_clips, embed_dim)``.
    """

    embed_dim: int
    space_id: str
    embed_fn: VideoEmbedFn
    name: str = "user"

    def __post_init__(self) -> None:
        _validate_common(self.embed_dim, self.space_id, self.name)


@dataclass
class TextEmbedder:
    """Text embedder (used to embed prompts for clip-level event detection).

    ``embed_fn`` receives ``list[str]`` and returns ``np.ndarray`` of
    shape ``(N, embed_dim)``.
    """

    embed_dim: int
    space_id: str
    embed_fn: TextEmbedFn
    name: str = "user"

    def __post_init__(self) -> None:
        _validate_common(self.embed_dim, self.space_id, self.name)


def _validate_common(embed_dim: int, space_id: str, name: str) -> None:
    if not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"embed_dim must be a positive int, got {embed_dim!r}")
    if not isinstance(space_id, str) or not space_id.strip():
        raise ValueError("space_id is required and must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")


# ---------------------------------------------------------------------------
# DummyEmbedder — deterministic, semantically meaningless, for tests
# ---------------------------------------------------------------------------


_DUMMY_EMBED_DIM = 128
_DUMMY_FEATURE_DIM = 16
_DUMMY_RNG = np.random.RandomState(42)
_DUMMY_PROJECTION: np.ndarray = _DUMMY_RNG.randn(_DUMMY_FEATURE_DIM, _DUMMY_EMBED_DIM).astype(np.float64)


def _frame_features(frame: Frame) -> np.ndarray:
    """Compute a tiny deterministic feature vector from a Frame.

    Channel means + stds + image dimensions + frame_id sentinel —
    enough to make a unique signature without any real semantics.
    """
    arr = np.asarray(frame.image, dtype=np.float64) / 255.0
    feats = np.empty(_DUMMY_FEATURE_DIM, dtype=np.float64)
    feats[0:3] = arr.reshape(-1, 3).mean(axis=0)
    feats[3:6] = arr.reshape(-1, 3).std(axis=0)
    feats[6:9] = arr.reshape(-1, 3).min(axis=0)
    feats[9:12] = arr.reshape(-1, 3).max(axis=0)
    feats[12] = float(frame.width) / 4096.0
    feats[13] = float(frame.height) / 4096.0
    feats[14] = float(frame.frame_id % 1024) / 1024.0
    feats[15] = float(frame.frame_id // 1024) / 1024.0
    return feats


def _dummy_image_embed(frames: list[Frame]) -> np.ndarray:
    if not frames:
        return np.zeros((0, _DUMMY_EMBED_DIM), dtype=np.float64)
    feats = np.stack([_frame_features(f) for f in frames], axis=0)
    return feats @ _DUMMY_PROJECTION


def _dummy_video_embed(clips: list[list[Frame]]) -> np.ndarray:
    if not clips:
        return np.zeros((0, _DUMMY_EMBED_DIM), dtype=np.float64)
    out = np.empty((len(clips), _DUMMY_EMBED_DIM), dtype=np.float64)
    for i, clip in enumerate(clips):
        if not clip:
            out[i] = 0.0
            continue
        per_frame = _dummy_image_embed(clip)
        out[i] = per_frame.mean(axis=0)
    return out


def _dummy_text_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _DUMMY_EMBED_DIM), dtype=np.float64)
    out = np.empty((len(texts), _DUMMY_EMBED_DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed_int = int.from_bytes(h[:8], "big") & 0xFFFFFFFF
        rng = np.random.RandomState(seed_int)
        out[i] = rng.randn(_DUMMY_EMBED_DIM)
    return out


class DummyEmbedder:
    """Factory helpers for fully-deterministic test embedders.

    All three roles share the same 128-d space (``spaces.DUMMY``) so
    they can be wired into a single Session as a clip/text pair.
    """

    embed_dim: int = _DUMMY_EMBED_DIM
    space_id: str = spaces.DUMMY

    @classmethod
    def make_image(cls, name: str = "dummy") -> ImageEmbedder:
        return ImageEmbedder(
            embed_dim=cls.embed_dim,
            space_id=cls.space_id,
            embed_fn=_dummy_image_embed,
            name=name,
        )

    @classmethod
    def make_video(cls, name: str = "dummy") -> VideoEmbedder:
        return VideoEmbedder(
            embed_dim=cls.embed_dim,
            space_id=cls.space_id,
            embed_fn=_dummy_video_embed,
            name=name,
        )

    @classmethod
    def make_text(cls, name: str = "dummy") -> TextEmbedder:
        return TextEmbedder(
            embed_dim=cls.embed_dim,
            space_id=cls.space_id,
            embed_fn=_dummy_text_embed,
            name=name,
        )


# ---------------------------------------------------------------------------
# ColorHistogramEmbedder — real frame features, 256-d, for `videospectra demo`
# ---------------------------------------------------------------------------


_HIST_BINS_PER_CHANNEL = 64
_HIST_GRAY_BINS = 64
_HIST_TOTAL_DIM = _HIST_BINS_PER_CHANNEL * 3 + _HIST_GRAY_BINS  # 192 + 64 = 256


def _histogram_one_frame(frame: Frame) -> np.ndarray:
    arr = np.asarray(frame.image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"ColorHistogramEmbedder expects RGB frames, got shape {arr.shape}")

    out = np.empty(_HIST_TOTAL_DIM, dtype=np.float64)
    offset = 0
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=_HIST_BINS_PER_CHANNEL, range=(0, 256))
        out[offset : offset + _HIST_BINS_PER_CHANNEL] = hist
        offset += _HIST_BINS_PER_CHANNEL

    # ITU-R BT.601 luminance (matches PIL.Image.convert('L'))
    gray = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114).astype(np.float64)
    gray_hist, _ = np.histogram(gray, bins=_HIST_GRAY_BINS, range=(0, 256))
    out[offset : offset + _HIST_GRAY_BINS] = gray_hist

    return out


def _color_hist_image_embed(frames: list[Frame]) -> np.ndarray:
    if not frames:
        return np.zeros((0, _HIST_TOTAL_DIM), dtype=np.float64)
    return np.stack([_histogram_one_frame(f) for f in frames], axis=0)


class ColorHistogramEmbedder:
    """Factory for a real, no-deps frame embedder.

    Output is a 256-d vector of per-channel RGB histograms (64 bins each)
    plus a 64-bin grayscale luminance histogram. This captures gross
    color/brightness changes — enough to drive entropy / motion /
    anomaly signals on a webcam — but is NOT a semantic embedder.

    Intended use: ``videospectra demo`` install-verification, smoke tests,
    no-GPU laptop development. Do not use for real video understanding.
    """

    embed_dim: int = _HIST_TOTAL_DIM
    space_id: str = spaces.COLOR_HISTOGRAM

    @classmethod
    def make_image(cls, name: str = "color-histogram") -> ImageEmbedder:
        return ImageEmbedder(
            embed_dim=cls.embed_dim,
            space_id=cls.space_id,
            embed_fn=_color_hist_image_embed,
            name=name,
        )
