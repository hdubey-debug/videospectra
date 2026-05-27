"""vnvideo setup file — full RzenEmbed (Qihoo360 Qwen2-VL) pipeline.

Reproduces the original von-neumann-dashboard demo end-to-end:

- 3584-d frame embeddings via ``RzenEmbed.get_image_embeddings``
- 3584-d clip embeddings via ``get_fused_embeddings(images=[[frames]])``
  with the video instruction
- 3584-d text embeddings via ``get_fused_embeddings(texts=[...])`` with
  the clip-text instruction
- 30-frame sliding window, 8-frame clips at 2 FPS (4 s per clip), all
  spectral signals enabled

Requirements:

- ~15 GB VRAM (Qwen2-VL backbone, bfloat16). Use a GPU node — NOT a
  Jetson Orin Nano. For lighter models on edge hardware, see
  ``open_clip_frame_pooling.py`` or write your own.
- RzenEmbed checked out at ``/mmfs1/.../rzenembed/RzenEmbed/``. We
  insert that path into sys.path on import.
- ``transformers``, ``torch`` (CUDA build), ``Pillow`` installed in
  the active environment.

Run::

    vnvideo serve --setup examples/rzenembed_full.py --port 8765
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import ImageEmbedder, TextEmbedder, VideoEmbedder
from vnvideo.session import ClipConfig, Session
from vnvideo.types import Frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match the original demo)
# ---------------------------------------------------------------------------

_RZEN_SRC_DEFAULT = Path("/mmfs1/scratch/jacks.local/hdubey/02-video-embeddings/rzenembed/RzenEmbed")
_RZEN_SPACE = "qihoo360/RzenEmbed@3584/video-retrieval"
_RZEN_DIM = 3584

CLIP_VIDEO_INSTRUCTION = "Understand the content of the provided video."
CLIP_TEXT_INSTRUCTION = "Find the video snippet that corresponds to the given caption: "

# ---------------------------------------------------------------------------
# Lazy model singleton — loaded once across all sessions
# ---------------------------------------------------------------------------

_MODEL = None


def _get_model():
    """Lazy-load the RzenEmbed model on first call. Idempotent."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    rzen_src = Path(__file__).resolve().parent.parent / "RzenEmbed"
    if not rzen_src.exists():
        rzen_src = _RZEN_SRC_DEFAULT
    if not rzen_src.exists():
        raise RuntimeError(
            f"RzenEmbed source not found at {rzen_src}. "
            "Check the path or install RzenEmbed beside this package."
        )
    sys.path.insert(0, str(rzen_src))

    # Imports are deferred so the example module can be loaded for
    # introspection (e.g., by tests) without pulling in torch/CUDA.
    from rzen_embed_inference import RzenEmbed  # noqa: PLC0415

    logger.info("Loading RzenEmbed (~30-60s) ...")
    model = RzenEmbed(
        "qihoo360/RzenEmbed",
        attn_implementation="sdpa",
        min_video_tokens=160,
        max_video_tokens=180,
        max_length=8000,
    )
    logger.info("RzenEmbed loaded")
    _MODEL = model
    return model


# ---------------------------------------------------------------------------
# Embed functions — sync, run in the Session's executor
# ---------------------------------------------------------------------------


def _embed_frames(frames: list[Frame]) -> np.ndarray:
    """Per-frame image embeddings.

    Matches ``_embed_frame`` in the original server.py: bfloat16
    tensor -> float32 -> numpy float64.
    """
    import torch  # noqa: PLC0415

    model = _get_model()
    pils = [f.image for f in frames]
    with torch.no_grad():
        out = model.get_image_embeddings(images=pils)
    return out.float().numpy().astype(np.float64)


def _embed_clips(clips: list[list[Frame]]) -> np.ndarray:
    """Clip-level video embeddings (one per clip)."""
    import torch  # noqa: PLC0415

    model = _get_model()
    pil_clips = [[f.image for f in clip] for clip in clips]
    embs = []
    for clip in pil_clips:
        with torch.no_grad():
            out = model.get_fused_embeddings(
                instruction=CLIP_VIDEO_INSTRUCTION,
                images=[clip],
            )
        embs.append(out[0].float().numpy().astype(np.float64))
    return np.stack(embs, axis=0)


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Text embeddings for clip-level event prompts."""
    import torch  # noqa: PLC0415

    model = _get_model()
    embs = []
    for text in texts:
        with torch.no_grad():
            out = model.get_fused_embeddings(
                instruction=CLIP_TEXT_INSTRUCTION,
                texts=[text],
            )
        embs.append(out[0].float().numpy().astype(np.float64))
    return np.stack(embs, axis=0)


# ---------------------------------------------------------------------------
# Session factory — the entry point vnvideo CLI calls
# ---------------------------------------------------------------------------


def make_session() -> Session:
    frame_embedder = ImageEmbedder(
        embed_dim=_RZEN_DIM,
        space_id=_RZEN_SPACE,
        embed_fn=_embed_frames,
        name="rzen-embed-frame",
    )
    clip_embedder = VideoEmbedder(
        embed_dim=_RZEN_DIM,
        space_id=_RZEN_SPACE,
        embed_fn=_embed_clips,
        name="rzen-embed-clip",
    )
    text_embedder = TextEmbedder(
        embed_dim=_RZEN_DIM,
        space_id=_RZEN_SPACE,
        embed_fn=_embed_texts,
        name="rzen-embed-text",
    )
    return Session(
        frame_embedder=frame_embedder,
        clip_embedder=clip_embedder,
        text_embedder=text_embedder,
        spectral_config=SpectralConfig(
            window_frames=30,
            top_k_eigen=30,
            top_k_vec=3,
            anomaly_threshold=0.5,
            recurrence_threshold_deg=15.0,
            min_recurrence_gap_frames=10,
            max_subspace_history=600,
            ema_alpha=0.3,
        ),
        clip_config=ClipConfig(clip_frames=8, clip_stride=8),
        # Serialize all calls into the single GPU.
        embedder_concurrency=1,
    )


if __name__ == "__main__":
    # Smoke check: can we build the session object? (Does NOT load the
    # model — that happens on first frame.)
    s = make_session()
    print(f"Session capabilities: {s.capabilities()}")
