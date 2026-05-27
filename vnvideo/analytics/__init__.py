"""Spectral analytics for video embeddings.

One cohesive module — invariant 7. No premature split into entropy /
motion / anomaly submodules.
"""
from __future__ import annotations

from vnvideo.analytics.spectral import (
    SpectralAnalyzer,
    SpectralConfig,
    SpectralUpdate,
)

__all__ = ["SpectralAnalyzer", "SpectralConfig", "SpectralUpdate"]
