"""Versioned Pydantic event schemas.

Every wire/disk event is a subclass of ``BaseEvent`` and serializes
into an envelope shaped like::

    {
      "schema_version": "0.1",
      "payload_version": "1.0",
      "type": "frame_metrics",
      "session_id": "ab12cd34",
      "source_id": "ws",
      "frame_id": 42,
      "monotonic_ts": 12345.678,
      "wall_ts": "2026-05-27T18:42:11.123456+00:00",
      "payload": { ... }
    }

The ``Event`` discriminated union parses any incoming JSON into the
correct subclass by reading the ``type`` field.

This module has zero FastAPI / websocket imports — it is core. The
server module wraps these into wire format; tests can construct them
without the server extra installed.
"""
# Pydantic resolves annotation strings at runtime via get_type_hints; on
# Python 3.9 the PEP 604 ``X | Y`` syntax fails to eval. We use the
# typing.Optional/Union forms here even though ruff prefers ``X | Y``,
# and disable the relevant ruff rules for this file in pyproject.toml.
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = "0.1"
PAYLOAD_VERSION = "1.0"


class _StrictModel(BaseModel):
    """Common config: forbid unknown fields so schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Envelope base
# ---------------------------------------------------------------------------


class BaseEvent(_StrictModel):
    schema_version: Literal["0.1"] = SCHEMA_VERSION
    payload_version: str = PAYLOAD_VERSION
    type: str
    session_id: str
    source_id: Optional[str] = None
    frame_id: Optional[int] = None
    monotonic_ts: float
    wall_ts: datetime = Field(default_factory=_now_utc)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


class FrameMetricsPayload(_StrictModel):
    entropy_norm: float
    motion_score: float
    anomaly_score: float
    buffer_fill: int
    infer_ms: float
    effective_rank: float


class ShotBoundaryPayload(_StrictModel):
    shot_count: int
    anomaly_at_cut: float


class AnomalyAlertPayload(_StrictModel):
    anomaly_score: float
    threshold: float
    sustained_frames: int


class RecurrencePayload(_StrictModel):
    angle_deg: float
    steps_ago: int
    seconds_ago: float


class ClipScore(_StrictModel):
    prompt_id: str
    text: str
    similarity: float


class ClipScoresPayload(_StrictModel):
    scores: list[ClipScore]
    clip_infer_ms: float


class PromptAddedPayload(_StrictModel):
    prompt_id: str
    text: str
    embed_ms: float


class PromptRemovedPayload(_StrictModel):
    prompt_id: str
    text: str


class FrameDroppedPayload(_StrictModel):
    frame_id: int
    reason: Literal["sink_overflow", "embedder_busy", "session_closing", "warmup_failed"]


class StatusPayload(_StrictModel):
    message: str
    level: Literal["info", "warning", "error"] = "info"


class EmbedderErrorPayload(_StrictModel):
    role: Literal["frame", "clip", "text"]
    error_type: str
    message: str


ConfigValue = Union[int, float, str, bool, None]


class SessionInfoPayload(_StrictModel):
    spectral_analytics: bool
    clip_events: bool
    prompts: bool
    embedder_names: dict[str, str]
    embedder_spaces: dict[str, str]
    config: dict[str, ConfigValue]


# ---------------------------------------------------------------------------
# Event subclasses (discriminated on ``type``)
# ---------------------------------------------------------------------------


class FrameMetrics(BaseEvent):
    type: Literal["frame_metrics"] = "frame_metrics"
    payload: FrameMetricsPayload


class ShotBoundary(BaseEvent):
    type: Literal["shot_boundary"] = "shot_boundary"
    payload: ShotBoundaryPayload


class AnomalyAlert(BaseEvent):
    type: Literal["anomaly_alert"] = "anomaly_alert"
    payload: AnomalyAlertPayload


class Recurrence(BaseEvent):
    type: Literal["recurrence"] = "recurrence"
    payload: RecurrencePayload


class ClipScores(BaseEvent):
    type: Literal["clip_scores"] = "clip_scores"
    payload: ClipScoresPayload


class PromptAdded(BaseEvent):
    type: Literal["prompt_added"] = "prompt_added"
    payload: PromptAddedPayload


class PromptRemoved(BaseEvent):
    type: Literal["prompt_removed"] = "prompt_removed"
    payload: PromptRemovedPayload


class FrameDropped(BaseEvent):
    type: Literal["frame_dropped"] = "frame_dropped"
    payload: FrameDroppedPayload


class Status(BaseEvent):
    type: Literal["status"] = "status"
    payload: StatusPayload


class EmbedderError(BaseEvent):
    type: Literal["embedder_error"] = "embedder_error"
    payload: EmbedderErrorPayload


class SessionInfo(BaseEvent):
    type: Literal["session_info"] = "session_info"
    payload: SessionInfoPayload


# ---------------------------------------------------------------------------
# Discriminated union for parsing
# ---------------------------------------------------------------------------


Event = Annotated[
    Union[
        FrameMetrics,
        ShotBoundary,
        AnomalyAlert,
        Recurrence,
        ClipScores,
        PromptAdded,
        PromptRemoved,
        FrameDropped,
        Status,
        EmbedderError,
        SessionInfo,
    ],
    Field(discriminator="type"),
]

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(data: Union[bytes, str, dict]) -> Event:
    """Parse a JSON string/bytes/dict into the correct Event subclass.

    Uses the discriminated union's ``type`` field to route. Raises
    ``pydantic.ValidationError`` on schema violations.
    """
    if isinstance(data, (bytes, str)):
        return _EVENT_ADAPTER.validate_json(data)
    return _EVENT_ADAPTER.validate_python(data)
