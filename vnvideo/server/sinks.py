"""WebSocket-attached sinks.

- ``WebSocketSink``: emits one JSON message per Event in the canonical
  versioned schema. The published API for new integrators.
- ``LegacyDashboardWebSocketSink``: aggregates events per frame_id and
  emits one flat per-frame message in the format the bundled poster
  dashboard expects. Frozen for backward compatibility; v0.2 will
  rewrite the dashboard against the new schema.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from vnvideo.events import (
    AnomalyAlert,
    ClipScores,
    EmbedderError,
    Event,
    FrameDropped,
    FrameMetrics,
    PromptAdded,
    PromptRemoved,
    Recurrence,
    SessionInfo,
    ShotBoundary,
    Status,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical sink — one message per Event, new schema
# ---------------------------------------------------------------------------


class WebSocketSink:
    """Send each event as a single JSON message over the WebSocket.

    The canonical wire format for new integrators. Used by the
    ``/ws/v1/events`` route.
    """

    name: str = "websocket"

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def emit(self, event: Event) -> None:
        # ``mode='json'`` makes datetime objects render as ISO strings
        await self._ws.send_json(event.model_dump(mode="json"))

    async def aclose(self) -> None:
        if self._ws.client_state != WebSocketState.DISCONNECTED:
            try:
                await self._ws.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("WebSocketSink close raised: %s", exc)


# ---------------------------------------------------------------------------
# Legacy sink — aggregates events into the old flat-per-frame format
# ---------------------------------------------------------------------------


class LegacyDashboardWebSocketSink:
    """Translate new-schema events into the v0 dashboard format.

    The bundled ``dashboard.html`` predates the versioned event schema
    and expects ONE message per frame with all derived fields flat:

        {status: "ok", entropy_norm, motion_score, anomaly_score,
         is_anomaly, is_shot_boundary, shot_count, recurrence,
         buffer_fill, infer_ms, clip_event_scores, clip_infer_ms,
         clip_progress, clip_total}

    Strategy: buffer per-frame derived events (ShotBoundary, Recurrence)
    by frame_id; flush as a combined message when FrameMetrics for the
    same frame_id arrives. FrameMetrics arrives LAST per frame by the
    Session emission order — see ``Session._process_frame_inner``.

    ClipScores arrive asynchronously (one per ~8 frames). The latest
    scores are cached and attached to every per-frame flush.

    Prompt events are translated 1:1.
    """

    name: str = "legacy-dashboard-ws"

    def __init__(self, ws: WebSocket, clip_total: int = 8) -> None:
        self._ws = ws
        self._clip_total = clip_total
        # Per-frame partial state keyed on frame_id (small TTL: cleared on flush)
        self._partial: dict[int, dict[str, Any]] = {}
        # Persistent "latest" state
        self._shot_count = 0
        self._clip_event_scores: dict[str, dict] = {}
        self._clip_infer_ms = 0.0
        self._clip_progress = 0

    async def emit(self, event: Event) -> None:
        if isinstance(event, SessionInfo):
            await self._send({"status": "ready", "msg": "Session ready"})
            return
        if isinstance(event, Status):
            level_to_status = {"info": "ready", "warning": "ready", "error": "error"}
            status = level_to_status.get(event.payload.level, "ready")
            await self._send({"status": status, "msg": event.payload.message})
            return
        if isinstance(event, FrameDropped):
            await self._send({"status": "dropped", "msg": "Frame dropped (backpressure)"})
            return
        if isinstance(event, PromptAdded):
            await self._send(
                {
                    "status": "clip_event_added",
                    "event_id": event.payload.prompt_id,
                    "text": event.payload.text,
                }
            )
            return
        if isinstance(event, PromptRemoved):
            await self._send(
                {
                    "status": "clip_event_removed",
                    "event_id": event.payload.prompt_id,
                }
            )
            return
        if isinstance(event, EmbedderError):
            await self._send(
                {
                    "status": "event_error",
                    "msg": f"{event.payload.role} embedder: {event.payload.message}",
                }
            )
            return

        # Per-frame events: buffer until FrameMetrics flushes.
        if isinstance(event, ShotBoundary):
            self._shot_count = event.payload.shot_count
            if event.frame_id is not None:
                self._partial.setdefault(event.frame_id, {})
                self._partial[event.frame_id]["is_shot_boundary"] = True
                self._partial[event.frame_id]["shot_count"] = event.payload.shot_count
            return
        if isinstance(event, Recurrence):
            if event.frame_id is not None:
                self._partial.setdefault(event.frame_id, {})
                self._partial[event.frame_id]["recurrence"] = {
                    "angle": round(event.payload.angle_deg, 1),
                    "steps_ago": event.payload.steps_ago,
                    "seconds_ago": event.payload.seconds_ago,
                }
            return
        if isinstance(event, AnomalyAlert):
            # The continuous score is already on FrameMetrics; the alert
            # itself just sets is_anomaly. No-op for the flat dict.
            return
        if isinstance(event, ClipScores):
            # Update cached snapshot — attached to every subsequent flush
            self._clip_event_scores = {
                s.prompt_id: {"text": s.text, "similarity": s.similarity}
                for s in event.payload.scores
            }
            self._clip_infer_ms = event.payload.clip_infer_ms
            return
        if isinstance(event, FrameMetrics):
            partial = self._partial.pop(event.frame_id or -1, {})
            flat = {
                "status": "ok",
                "entropy_norm": event.payload.entropy_norm,
                "motion_score": event.payload.motion_score,
                "anomaly_score": event.payload.anomaly_score,
                "is_anomaly": event.payload.anomaly_score > 0.5,
                "is_shot_boundary": partial.get("is_shot_boundary", False),
                "shot_count": partial.get("shot_count", self._shot_count),
                "recurrence": partial.get("recurrence"),
                "buffer_fill": event.payload.buffer_fill,
                "infer_ms": round(event.payload.infer_ms, 1),
                "clip_event_scores": self._clip_event_scores,
                "clip_infer_ms": round(self._clip_infer_ms, 1),
                "clip_progress": self._clip_progress,
                "clip_total": self._clip_total,
            }
            await self._send(flat)
            return

        # Unknown event — silently ignore (forward compatibility)
        logger.debug("legacy-dashboard sink ignoring unknown event type: %s", type(event).__name__)

    async def _send(self, payload: dict) -> None:
        await self._ws.send_json(payload)

    async def aclose(self) -> None:
        if self._ws.client_state != WebSocketState.DISCONNECTED:
            try:
                await self._ws.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("legacy ws sink close raised: %s", exc)
