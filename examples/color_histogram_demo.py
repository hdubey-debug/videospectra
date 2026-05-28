"""Minimal videospectra setup file — frame-only ColorHistogram embedder.

This is the SAME session ``videospectra demo`` builds internally; it exists
as a standalone example so users can see the shape of a setup file
they would write themselves.

Run::

    videospectra serve --setup examples/color_histogram_demo.py

Open http://127.0.0.1:8765 in a browser, click START to enable the
webcam, and watch entropy / motion / anomaly traces respond to scene
changes.

This is for install verification, not real video understanding. The
color histogram has no semantic content beyond color/brightness.
"""
from __future__ import annotations

from videospectra.analytics.spectral import SpectralConfig
from videospectra.embedders import ColorHistogramEmbedder
from videospectra.session import Session


def make_session() -> Session:
    return Session(
        frame_embedder=ColorHistogramEmbedder.make_image(),
        spectral_config=SpectralConfig(window_frames=30),
    )
