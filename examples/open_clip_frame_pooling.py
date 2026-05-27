"""vnvideo setup file — open_clip image/text + frame-pooling for clips.

Uses the stock OpenCLIP ViT-L-14 image and text encoders. Clip
embeddings are produced by averaging per-frame image embeddings —
this is a baseline (frame pooling), NOT a real video model. Use it
for laptop / no-GPU experiments where you want a working "clip vs
prompt" similarity even though the model has no temporal awareness.

Requirements::

    pip install open_clip_torch torch

(``open_clip_torch`` is NOT a vnvideo dependency — install it
yourself.)

Run::

    vnvideo serve --setup examples/open_clip_frame_pooling.py
"""
from __future__ import annotations

import logging

import numpy as np

from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import ImageEmbedder, TextEmbedder, VideoEmbedder
from vnvideo.session import ClipConfig, Session
from vnvideo.types import Frame

logger = logging.getLogger(__name__)

_OPEN_CLIP_MODEL = "ViT-L-14"
_OPEN_CLIP_PRETRAINED = "openai"
_OPEN_CLIP_DIM = 768
_OPEN_CLIP_SPACE = f"open_clip/{_OPEN_CLIP_MODEL}@{_OPEN_CLIP_DIM}/image-text"

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None


def _get_model_and_preprocess():
    global _MODEL, _PREPROCESS, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _TOKENIZER

    try:
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_frame_pooling requires `pip install open_clip_torch torch`"
        ) from exc

    logger.info("Loading OpenCLIP %s (%s)", _OPEN_CLIP_MODEL, _OPEN_CLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        _OPEN_CLIP_MODEL, pretrained=_OPEN_CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(_OPEN_CLIP_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    _MODEL, _PREPROCESS, _TOKENIZER = model, preprocess, tokenizer
    return _MODEL, _PREPROCESS, _TOKENIZER


def _embed_frames(frames: list[Frame]) -> np.ndarray:
    import torch  # noqa: PLC0415

    model, preprocess, _ = _get_model_and_preprocess()
    device = next(model.parameters()).device
    batch = torch.stack([preprocess(f.image) for f in frames]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
    return feats.float().cpu().numpy().astype(np.float64)


def _embed_clips(clips: list[list[Frame]]) -> np.ndarray:
    """Frame-pool clip embeddings: average per-frame image embeddings.

    Honest naming: this is NOT real video understanding. A frame-pooled
    embedding ignores temporal dynamics, ordering, and event structure.
    Use a true video model (e.g., RzenEmbed) for those.
    """
    out = np.empty((len(clips), _OPEN_CLIP_DIM), dtype=np.float64)
    for i, clip in enumerate(clips):
        frame_embs = _embed_frames(clip)
        out[i] = frame_embs.mean(axis=0)
    return out


def _embed_texts(texts: list[str]) -> np.ndarray:
    import torch  # noqa: PLC0415

    model, _, tokenizer = _get_model_and_preprocess()
    device = next(model.parameters()).device
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return feats.float().cpu().numpy().astype(np.float64)


def make_session() -> Session:
    frame_emb = ImageEmbedder(
        embed_dim=_OPEN_CLIP_DIM,
        space_id=_OPEN_CLIP_SPACE,
        embed_fn=_embed_frames,
        name=f"open_clip-{_OPEN_CLIP_MODEL}",
    )
    clip_emb = VideoEmbedder(
        embed_dim=_OPEN_CLIP_DIM,
        space_id=_OPEN_CLIP_SPACE,
        embed_fn=_embed_clips,
        name=f"open_clip-{_OPEN_CLIP_MODEL}-pooled",
    )
    text_emb = TextEmbedder(
        embed_dim=_OPEN_CLIP_DIM,
        space_id=_OPEN_CLIP_SPACE,
        embed_fn=_embed_texts,
        name=f"open_clip-{_OPEN_CLIP_MODEL}-text",
    )
    return Session(
        frame_embedder=frame_emb,
        clip_embedder=clip_emb,
        text_embedder=text_emb,
        spectral_config=SpectralConfig(),
        clip_config=ClipConfig(clip_frames=8, clip_stride=8),
        embedder_concurrency=1,
    )
