"""FastAPI app factory.

``create_app(session_factory)`` builds an app whose WebSocket and HTTP
routes spin up a Session via the supplied factory. The factory is
called on each WebSocket connection (one Session per connection).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vnvideo import __version__
from vnvideo.server.routes import register_websocket_routes
from vnvideo.session import Session

STATIC_DIR = Path(__file__).parent / "static"


def create_app(session_factory: Callable[[], Session]) -> FastAPI:
    """Build a FastAPI app whose WebSocket route uses ``session_factory``.

    The factory is invoked per connection. Each connection gets its
    own Session instance (and its own SinkRunners, embedder semaphore,
    etc.). For multi-tenant / multi-camera scaling, this design is the
    natural extension point — but v0.1 ships single-session-per-WS only.
    """
    app = FastAPI(title="vnvideo", version=__version__)

    # Static files: dashboard.html, plotly.min.js
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard() -> HTMLResponse:
        html_path = STATIC_DIR / "dashboard.html"
        if not html_path.exists():
            return HTMLResponse(
                "<h1>vnvideo</h1><p>Dashboard not bundled in this install. "
                "See /api/v1/info for capabilities.</p>",
                status_code=200,
            )
        return HTMLResponse(html_path.read_text())

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"model_ready": True, "version": __version__})

    @app.get("/api/v1/info")
    async def api_info() -> JSONResponse:
        # A separate ephemeral Session just to report capabilities
        try:
            sample = session_factory()
            caps = sample.capabilities()
            await sample.aclose()
            payload = {
                "version": __version__,
                "schema_version": "0.1",
                "capabilities": {
                    "spectral_analytics": caps.spectral_analytics,
                    "clip_events": caps.clip_events,
                    "prompts": caps.prompts,
                    "embedder_names": caps.embedder_names,
                    "embedder_spaces": caps.embedder_spaces,
                },
            }
        except Exception as exc:  # noqa: BLE001
            payload = {"version": __version__, "error": str(exc)}
        return JSONResponse(payload)

    register_websocket_routes(app, session_factory)
    return app
