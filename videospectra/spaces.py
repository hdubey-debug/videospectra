"""Canonical ``space_id`` constants for videospectra built-in embedders.

A ``space_id`` is opaque, required, and used only for equality checks
(specifically: ``clip_embedder.space_id == text_embedder.space_id``).
The convention used by built-ins is ``"<vendor>/<model>@<dim>/<task>"``,
but the framework treats it as an opaque string.
"""
from __future__ import annotations

DUMMY: str = "videospectra/dummy@128/test"
COLOR_HISTOGRAM: str = "videospectra/color-histogram@256/raw"
