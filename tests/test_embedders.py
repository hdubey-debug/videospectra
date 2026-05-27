"""Tests for vnvideo.embedders and vnvideo._internal.normalize."""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
from PIL import Image

from vnvideo import embedders as emb_mod
from vnvideo import spaces
from vnvideo._internal.normalize import l2_normalize
from vnvideo.embedders import (
    ColorHistogramEmbedder,
    DummyEmbedder,
    ImageEmbedder,
    TextEmbedder,
    VideoEmbedder,
)
from vnvideo.types import Frame


def _make_frame(frame_id: int = 0, color: tuple[int, int, int] = (128, 64, 32), size: tuple[int, int] = (32, 32)) -> Frame:
    return Frame.from_pil(Image.new("RGB", size, color), source_id="t", frame_id=frame_id)


class TestL2Normalize:
    def test_1d_unit_norm(self) -> None:
        v = np.array([3.0, 4.0])
        out = l2_normalize(v)
        assert np.isclose(np.linalg.norm(out), 1.0)
        assert np.allclose(out, [0.6, 0.8])

    def test_2d_per_row(self) -> None:
        m = np.array([[3.0, 4.0], [0.0, 1.0], [1.0, 0.0]])
        out = l2_normalize(m)
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0)

    def test_zero_vector_remains_zero(self) -> None:
        v = np.zeros(8)
        out = l2_normalize(v)
        assert np.allclose(out, 0.0)

    def test_idempotent_on_unit_vector(self) -> None:
        v = np.array([0.6, 0.8])
        out = l2_normalize(v)
        out2 = l2_normalize(out)
        assert np.allclose(out, out2)

    def test_does_not_mutate_input(self) -> None:
        v = np.array([3.0, 4.0])
        original = v.copy()
        _ = l2_normalize(v)
        assert np.array_equal(v, original)

    def test_3d_input_raises(self) -> None:
        v = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="1-D or 2-D"):
            l2_normalize(v)


class TestEmbedderValidation:
    def test_empty_space_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="space_id"):
            ImageEmbedder(embed_dim=8, space_id="", embed_fn=lambda f: np.zeros((len(f), 8)))

    def test_whitespace_space_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="space_id"):
            ImageEmbedder(embed_dim=8, space_id="   ", embed_fn=lambda f: np.zeros((len(f), 8)))

    def test_zero_embed_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            ImageEmbedder(embed_dim=0, space_id="x", embed_fn=lambda f: np.zeros((len(f), 8)))

    def test_negative_embed_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            VideoEmbedder(embed_dim=-1, space_id="x", embed_fn=lambda c: np.zeros((len(c), 8)))

    def test_text_embedder_validates_too(self) -> None:
        with pytest.raises(ValueError, match="space_id"):
            TextEmbedder(embed_dim=8, space_id="", embed_fn=lambda t: np.zeros((len(t), 8)))

    def test_no_normalized_field(self) -> None:
        # Catch accidental re-introduction of the normalized footgun.
        for cls in (ImageEmbedder, VideoEmbedder, TextEmbedder):
            names = {f.name for f in fields(cls)}
            assert "normalized" not in names, f"{cls.__name__} must not have a normalized field"


class TestDummyEmbedder:
    def test_image_shape(self) -> None:
        emb = DummyEmbedder.make_image()
        frames = [_make_frame(i) for i in range(5)]
        out = emb.embed_fn(frames)
        assert out.shape == (5, emb.embed_dim)

    def test_image_deterministic(self) -> None:
        emb = DummyEmbedder.make_image()
        frames = [_make_frame(i) for i in range(3)]
        out1 = emb.embed_fn(frames)
        out2 = emb.embed_fn(frames)
        assert np.array_equal(out1, out2)

    def test_image_different_frames_different_outputs(self) -> None:
        emb = DummyEmbedder.make_image()
        f1 = _make_frame(0, color=(255, 0, 0))
        f2 = _make_frame(1, color=(0, 255, 0))
        out = emb.embed_fn([f1, f2])
        assert not np.allclose(out[0], out[1])

    def test_video_shape(self) -> None:
        emb = DummyEmbedder.make_video()
        clips = [[_make_frame(i) for i in range(4)] for _ in range(3)]
        out = emb.embed_fn(clips)
        assert out.shape == (3, emb.embed_dim)

    def test_text_shape_and_determinism(self) -> None:
        emb = DummyEmbedder.make_text()
        out1 = emb.embed_fn(["a", "b", "a"])
        out2 = emb.embed_fn(["a", "b", "a"])
        assert out1.shape == (3, emb.embed_dim)
        assert np.array_equal(out1, out2)
        # Same text -> same vector
        assert np.allclose(out1[0], out1[2])
        # Different text -> different vector
        assert not np.allclose(out1[0], out1[1])

    def test_empty_input_is_safe(self) -> None:
        img = DummyEmbedder.make_image()
        vid = DummyEmbedder.make_video()
        txt = DummyEmbedder.make_text()
        assert img.embed_fn([]).shape == (0, img.embed_dim)
        assert vid.embed_fn([]).shape == (0, vid.embed_dim)
        assert txt.embed_fn([]).shape == (0, txt.embed_dim)

    def test_all_three_share_space(self) -> None:
        assert DummyEmbedder.make_image().space_id == spaces.DUMMY
        assert DummyEmbedder.make_video().space_id == spaces.DUMMY
        assert DummyEmbedder.make_text().space_id == spaces.DUMMY


class TestColorHistogramEmbedder:
    def test_shape_is_256(self) -> None:
        emb = ColorHistogramEmbedder.make_image()
        assert emb.embed_dim == 256
        frames = [_make_frame(i, color=(i * 30 % 256, 64, 32)) for i in range(4)]
        out = emb.embed_fn(frames)
        assert out.shape == (4, 256)

    def test_deterministic(self) -> None:
        emb = ColorHistogramEmbedder.make_image()
        frames = [_make_frame(0, color=(100, 200, 50))]
        out1 = emb.embed_fn(frames)
        out2 = emb.embed_fn(frames)
        assert np.array_equal(out1, out2)

    def test_different_colors_give_different_outputs(self) -> None:
        emb = ColorHistogramEmbedder.make_image()
        f1 = _make_frame(0, color=(255, 0, 0))
        f2 = _make_frame(1, color=(0, 0, 255))
        out = emb.embed_fn([f1, f2])
        assert not np.allclose(out[0], out[1])

    def test_constant_image_concentrates_mass(self) -> None:
        # A solid-color image puts all pixels into one bin per channel,
        # which after normalization gives a high-magnitude vector.
        emb = ColorHistogramEmbedder.make_image()
        out = emb.embed_fn([_make_frame(0, color=(128, 128, 128), size=(32, 32))])
        normed = l2_normalize(out[0])
        # Mass should concentrate at specific bins (the channel value 128 -> bin 32 of 64)
        # so the resulting vector has a small number of large peaks
        assert (normed > 0.05).sum() <= 8

    def test_space_id(self) -> None:
        emb = ColorHistogramEmbedder.make_image()
        assert emb.space_id == spaces.COLOR_HISTOGRAM


class TestNoTorchImport:
    def test_embedders_module_does_not_import_torch(self) -> None:
        # ColorHistogram and Dummy must be importable without torch installed.
        # We check the source file directly — sys.modules is unreliable
        # because rzen_venv has torch loaded from other modules.
        assert not any("torch" in name for name in dir(emb_mod))
        with open(emb_mod.__file__) as f:
            source = f.read()
        assert "import torch" not in source
        assert "from torch" not in source
