"""FastAPI adapter — thin transport over Session.

This package is the only place in vnvideo that imports FastAPI.
Importing ``vnvideo.server`` triggers the FastAPI import; users
without the ``[server]`` extra installed can still use the rest of
the package.
"""
from __future__ import annotations

from vnvideo.server.app import create_app
from vnvideo.server.sinks import LegacyDashboardWebSocketSink, WebSocketSink

__all__ = ["create_app", "WebSocketSink", "LegacyDashboardWebSocketSink"]
