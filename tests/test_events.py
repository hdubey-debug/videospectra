"""Tests for vnvideo.events — discriminated union, JSON roundtrip, schema."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vnvideo import events as events_mod
from vnvideo.events import (
    PAYLOAD_VERSION,
    SCHEMA_VERSION,
    AnomalyAlert,
    AnomalyAlertPayload,
    ClipScore,
    ClipScores,
    ClipScoresPayload,
    EmbedderError,
    EmbedderErrorPayload,
    FrameDropped,
    FrameDroppedPayload,
    FrameMetrics,
    FrameMetricsPayload,
    PromptAdded,
    PromptAddedPayload,
    PromptRemoved,
    PromptRemovedPayload,
    Recurrence,
    RecurrencePayload,
    SessionInfo,
    SessionInfoPayload,
    ShotBoundary,
    ShotBoundaryPayload,
    Status,
    StatusPayload,
    parse_event,
)


def _envelope_defaults() -> dict:
    return {
        "session_id": "abc12345",
        "source_id": "ws",
        "frame_id": 0,
        "monotonic_ts": 1234.567,
        "wall_ts": datetime(2026, 5, 27, 18, 42, 11, tzinfo=timezone.utc),
    }


class TestEnvelope:
    def test_schema_version_locked(self) -> None:
        e = FrameMetrics(
            **_envelope_defaults(),
            payload=FrameMetricsPayload(
                entropy_norm=0.5,
                motion_score=0.1,
                anomaly_score=0.05,
                buffer_fill=30,
                infer_ms=12.3,
                effective_rank=4.5,
            ),
        )
        assert e.schema_version == "0.1"
        assert e.payload_version == "1.0"
        assert SCHEMA_VERSION == "0.1"
        assert PAYLOAD_VERSION == "1.0"

    def test_wall_ts_serializes_to_iso(self) -> None:
        e = Status(**_envelope_defaults(), payload=StatusPayload(message="hi"))
        d = e.model_dump(mode="json")
        assert isinstance(d["wall_ts"], str)
        assert "2026-05-27T18:42:11" in d["wall_ts"]

    def test_default_wall_ts_is_utc(self) -> None:
        defaults = _envelope_defaults()
        del defaults["wall_ts"]
        e = Status(**defaults, payload=StatusPayload(message="hi"))
        assert e.wall_ts.tzinfo is not None

    def test_frozen(self) -> None:
        e = Status(**_envelope_defaults(), payload=StatusPayload(message="hi"))
        with pytest.raises(ValidationError):
            e.session_id = "different"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FrameMetrics(
                **_envelope_defaults(),
                payload=FrameMetricsPayload(
                    entropy_norm=0.5,
                    motion_score=0.1,
                    anomaly_score=0.05,
                    buffer_fill=30,
                    infer_ms=12.3,
                    effective_rank=4.5,
                ),
                unknown_field="x",
            )

    def test_source_id_and_frame_id_optional(self) -> None:
        defaults = _envelope_defaults()
        defaults["source_id"] = None
        defaults["frame_id"] = None
        e = Status(**defaults, payload=StatusPayload(message="hi"))
        assert e.source_id is None
        assert e.frame_id is None


class TestRoundtrip:
    """Every event subclass roundtrips through JSON correctly."""

    @pytest.mark.parametrize(
        "event_cls, payload_cls, payload_kwargs",
        [
            (
                FrameMetrics,
                FrameMetricsPayload,
                dict(
                    entropy_norm=0.42,
                    motion_score=0.13,
                    anomaly_score=0.07,
                    buffer_fill=30,
                    infer_ms=12.3,
                    effective_rank=4.8,
                ),
            ),
            (ShotBoundary, ShotBoundaryPayload, dict(shot_count=3, anomaly_at_cut=0.7)),
            (
                AnomalyAlert,
                AnomalyAlertPayload,
                dict(anomaly_score=0.85, threshold=0.5, sustained_frames=6),
            ),
            (
                Recurrence,
                RecurrencePayload,
                dict(angle_deg=8.2, steps_ago=45, seconds_ago=22.5),
            ),
            (
                ClipScores,
                ClipScoresPayload,
                dict(
                    scores=[
                        ClipScore(prompt_id="p1", text="a dog", similarity=0.6),
                        ClipScore(prompt_id="p2", text="a cat", similarity=0.4),
                    ],
                    clip_infer_ms=180.0,
                ),
            ),
            (
                PromptAdded,
                PromptAddedPayload,
                dict(prompt_id="p1", text="a dog", embed_ms=24.0),
            ),
            (
                PromptRemoved,
                PromptRemovedPayload,
                dict(prompt_id="p1", text="a dog"),
            ),
            (
                FrameDropped,
                FrameDroppedPayload,
                dict(frame_id=12, reason="sink_overflow"),
            ),
            (Status, StatusPayload, dict(message="up", level="info")),
            (
                EmbedderError,
                EmbedderErrorPayload,
                dict(role="frame", error_type="ValueError", message="bad shape"),
            ),
            (
                SessionInfo,
                SessionInfoPayload,
                dict(
                    spectral_analytics=True,
                    clip_events=True,
                    prompts=True,
                    embedder_names={"frame": "rzen"},
                    embedder_spaces={"frame": "qihoo/rzen@3584/video"},
                    config={"window_frames": 30, "clip_frames": 8},
                ),
            ),
        ],
    )
    def test_roundtrip_via_json(self, event_cls, payload_cls, payload_kwargs) -> None:
        e = event_cls(**_envelope_defaults(), payload=payload_cls(**payload_kwargs))
        s = e.model_dump_json()
        parsed = parse_event(s)
        assert isinstance(parsed, event_cls)
        assert parsed == e


class TestDiscriminator:
    def test_dispatches_on_type(self) -> None:
        d = {
            "schema_version": "0.1",
            "payload_version": "1.0",
            "type": "anomaly_alert",
            "session_id": "abc",
            "source_id": None,
            "frame_id": None,
            "monotonic_ts": 0.0,
            "wall_ts": "2026-05-27T18:42:11+00:00",
            "payload": {
                "anomaly_score": 0.9,
                "threshold": 0.5,
                "sustained_frames": 6,
            },
        }
        e = parse_event(d)
        assert isinstance(e, AnomalyAlert)
        assert e.payload.anomaly_score == 0.9

    def test_unknown_type_rejected(self) -> None:
        d = {
            "schema_version": "0.1",
            "payload_version": "1.0",
            "type": "unknown_type",
            "session_id": "abc",
            "monotonic_ts": 0.0,
            "wall_ts": "2026-05-27T18:42:11+00:00",
            "payload": {},
        }
        with pytest.raises(ValidationError):
            parse_event(d)

    def test_missing_type_rejected(self) -> None:
        d = {
            "schema_version": "0.1",
            "session_id": "abc",
            "monotonic_ts": 0.0,
            "wall_ts": "2026-05-27T18:42:11+00:00",
        }
        with pytest.raises(ValidationError):
            parse_event(d)

    def test_parse_from_bytes(self) -> None:
        e = Status(**_envelope_defaults(), payload=StatusPayload(message="hi"))
        b = e.model_dump_json().encode("utf-8")
        parsed = parse_event(b)
        assert isinstance(parsed, Status)


class TestPayloadShapes:
    def test_frame_dropped_reason_enum(self) -> None:
        with pytest.raises(ValidationError):
            FrameDroppedPayload(frame_id=0, reason="bogus")  # type: ignore[arg-type]

    def test_status_level_enum(self) -> None:
        with pytest.raises(ValidationError):
            StatusPayload(message="x", level="critical")  # type: ignore[arg-type]

    def test_embedder_error_role_enum(self) -> None:
        with pytest.raises(ValidationError):
            EmbedderErrorPayload(role="audio", error_type="ValueError", message="x")  # type: ignore[arg-type]

    def test_clip_scores_can_be_empty(self) -> None:
        p = ClipScoresPayload(scores=[], clip_infer_ms=1.0)
        assert p.scores == []


class TestFrameMetricsVsAnomalyAlert:
    """The continuous anomaly_score lives on FrameMetrics; AnomalyAlert
    is the discrete event the Session emits when a sustained run crosses
    the threshold. They must remain distinct."""

    def test_frame_metrics_has_continuous_anomaly_score(self) -> None:
        # FrameMetricsPayload contains the continuous score
        assert "anomaly_score" in FrameMetricsPayload.model_fields

    def test_anomaly_alert_has_threshold_and_sustained_frames(self) -> None:
        # AnomalyAlertPayload distinguishes the discrete event with extra context
        assert "threshold" in AnomalyAlertPayload.model_fields
        assert "sustained_frames" in AnomalyAlertPayload.model_fields

    def test_shot_boundary_has_anomaly_at_cut_not_is_anomaly(self) -> None:
        # Per invariant: ShotBoundary fires on rising edge in Session,
        # so the schema captures the value at the cut, not a flag.
        assert "anomaly_at_cut" in ShotBoundaryPayload.model_fields
        assert "is_anomaly" not in ShotBoundaryPayload.model_fields


class TestNoFastapiImport:
    def test_events_module_does_not_import_fastapi(self) -> None:
        with open(events_mod.__file__) as f:
            source = f.read()
        assert "import fastapi" not in source
        assert "from fastapi" not in source
