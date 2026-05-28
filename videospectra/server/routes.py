"""WebSocket route handlers.

Two WS routes:

- ``/ws`` — bundled dashboard ('legacy-dashboard' format)
- ``/ws/v1/events`` — canonical new-schema events

Both consume incoming binary messages as JPEG frames and incoming text
messages as JSON commands. The command vocabulary is the legacy
dashboard's: ``add_clip_event`` / ``remove_clip_event`` / ``list``. The
new-schema route also accepts ``add_prompt`` / ``remove_prompt`` as
canonical aliases.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from videospectra.events import Status, StatusPayload
from videospectra.server.sinks import LegacyDashboardWebSocketSink, WebSocketSink
from videospectra.session import EmbedderValidationError, Session
from videospectra.types import Frame

logger = logging.getLogger(__name__)


def register_websocket_routes(
    app: FastAPI,
    session_factory: Callable[[], Session],
) -> None:
    @app.websocket("/ws")
    async def ws_legacy(ws: WebSocket) -> None:
        """Bundled dashboard endpoint — emits legacy aggregated format."""
        await _run_ws_session(
            ws,
            session_factory,
            sink_factory=lambda websocket, session: LegacyDashboardWebSocketSink(
                websocket,
                clip_total=(
                    session._resolved_clip.clip_frames
                    if session._resolved_clip is not None
                    else 8
                ),
                anomaly_threshold=(
                    session.spectral_config.anomaly_threshold
                    if session.spectral_config is not None
                    else 0.5
                ),
            ),
        )

    @app.websocket("/ws/v1/events")
    async def ws_events(ws: WebSocket) -> None:
        """Canonical event-stream endpoint — emits new versioned schema."""
        await _run_ws_session(
            ws,
            session_factory,
            sink_factory=lambda websocket, _session: WebSocketSink(websocket),
        )


async def _run_ws_session(
    ws: WebSocket,
    session_factory: Callable[[], Session],
    *,
    sink_factory: Callable[[WebSocket, Session], object],
) -> None:
    await ws.accept()
    logger.info("WebSocket connected: %s", ws.client)

    session = session_factory()
    sink = sink_factory(ws, session)
    await session.add_sink(sink)  # type: ignore[arg-type]
    await session.start()

    try:
        await session.warmup()
    except EmbedderValidationError as exc:
        await ws.send_json(
            {"status": "error", "msg": f"warmup failed: {exc}"}
        )
        await session.aclose()
        await ws.close()
        return
    except Exception as exc:  # noqa: BLE001
        await ws.send_json(
            {"status": "error", "msg": f"warmup raised: {exc}"}
        )
        await session.aclose()
        await ws.close()
        return

    frame_id = 0
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                data = message["bytes"]
                try:
                    frame = Frame.from_jpeg(data, source_id="ws", frame_id=frame_id)
                    frame_id += 1
                except Exception as exc:  # noqa: BLE001
                    await ws.send_json({"status": "error", "msg": f"bad JPEG: {exc}"})
                    continue
                await session.process_frame(frame)
                continue

            if "text" in message and message["text"] is not None:
                try:
                    cmd = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"status": "error", "msg": "invalid JSON command"})
                    continue
                await _dispatch_command(session, cmd, ws)
                continue
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:  # noqa: BLE001
        logger.exception("WebSocket loop error: %s", exc)
        await _safe_emit_status(session, f"server error: {exc}", level="error")
    finally:
        await session.aclose()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


async def _dispatch_command(session: Session, cmd: dict, ws: WebSocket) -> None:
    """Handle text commands on the WebSocket.

    Accepts both legacy ('add_clip_event' / 'remove_clip_event' /
    'list_clip_events') and canonical ('add_prompt' / 'remove_prompt' /
    'list_prompts') vocabularies.
    """
    action = cmd.get("action")

    if action in ("add_clip_event", "add_prompt"):
        text = cmd.get("text", "")
        if not text:
            await ws.send_json({"status": "error", "msg": "empty prompt text"})
            return
        try:
            await session.add_prompt(text)
        except Exception as exc:  # noqa: BLE001
            await ws.send_json({"status": "error", "msg": f"add_prompt failed: {exc}"})
        return

    if action in ("remove_clip_event", "remove_prompt"):
        prompt_id = cmd.get("event_id") or cmd.get("prompt_id")
        if not prompt_id:
            await ws.send_json({"status": "error", "msg": "missing prompt_id"})
            return
        try:
            await session.remove_prompt(prompt_id)
        except Exception as exc:  # noqa: BLE001
            await ws.send_json({"status": "error", "msg": f"remove_prompt failed: {exc}"})
        return

    if action in ("list_clip_events", "list_prompts"):
        await ws.send_json({"status": "event_list", "events": session.list_prompts()})
        return

    await ws.send_json({"status": "error", "msg": f"unknown action: {action}"})


async def _safe_emit_status(session: Session, message: str, *, level: str) -> None:
    try:
        await session._emit(
            Status(
                session_id=session.session_id,
                source_id=None,
                frame_id=None,
                monotonic_ts=time.perf_counter(),
                wall_ts=datetime.now(timezone.utc),
                payload=StatusPayload(message=message, level=level),  # type: ignore[arg-type]
            )
        )
    except Exception:  # noqa: BLE001
        pass
