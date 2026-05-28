"""FastAPI adapter — thin transport over Session.

This package is the only place in videospectra that imports FastAPI.
Importing ``videospectra.server`` triggers the FastAPI import; users
without the ``[server]`` extra installed can still use the rest of
the package.
"""
from __future__ import annotations

from videospectra.server.app import create_app
from videospectra.server.sinks import LegacyDashboardWebSocketSink, WebSocketSink

__all__ = ["create_app", "WebSocketSink", "LegacyDashboardWebSocketSink"]
