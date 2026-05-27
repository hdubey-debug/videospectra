"""End-to-end integration: HTTP + WebSocket against a real server.

Uses ``fastapi.testclient.TestClient`` (which runs the app in-process
over a starlette test transport). No model loading required — uses
``DummyEmbedder`` / ``ColorHistogramEmbedder``.
"""
from __future__ import annotations

import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import ColorHistogramEmbedder, DummyEmbedder, ImageEmbedder
from vnvideo.events import parse_event
from vnvideo.server import create_app
from vnvideo.session import ClipConfig, Session


def _make_jpeg(seed: int = 0) -> bytes:
    img = Image.new("RGB", (32, 32), ((seed * 7) % 256, (seed * 13) % 256, (seed * 19) % 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _build_demo_session() -> Session:
    return Session(
        frame_embedder=ColorHistogramEmbedder.make_image(),
        spectral_config=SpectralConfig(window_frames=10),
    )


def _build_full_session() -> Session:
    return Session(
        frame_embedder=DummyEmbedder.make_image(),
        clip_embedder=DummyEmbedder.make_video(),
        text_embedder=DummyEmbedder.make_text(),
        spectral_config=SpectralConfig(window_frames=5),
        clip_config=ClipConfig(clip_frames=4, clip_stride=4),
        embedder_concurrency=2,
    )


class TestEndToEndDemo:
    def test_30_frames_through_legacy_ws(self) -> None:
        """30 fake JPEGs into /ws should produce 30 'ok' status messages."""
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                _ = ws.receive_json()  # ready
                for i in range(30):
                    ws.send_bytes(_make_jpeg(i))
                ok_count = 0
                for _ in range(60):  # generous polling — some may be other status
                    msg = ws.receive_json()
                    if msg.get("status") == "ok":
                        ok_count += 1
                    if ok_count == 30:
                        break
                assert ok_count == 30

    def test_30_frames_through_new_ws(self) -> None:
        """Every emitted message on /ws/v1/events should parse via Pydantic discriminated union."""
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/v1/events") as ws:
                first = ws.receive_json()
                e = parse_event(first)
                assert e.type == "session_info"

                # Send 10 frames so we're guaranteed at least 10 FrameMetrics events
                for i in range(10):
                    ws.send_bytes(_make_jpeg(i))
                seen_metric = False
                for _ in range(10):
                    msg = ws.receive_json()
                    e = parse_event(msg)
                    assert e.schema_version == "0.1"
                    assert e.payload_version == "1.0"
                    assert e.type in {
                        "session_info",
                        "frame_metrics",
                        "shot_boundary",
                        "anomaly_alert",
                        "recurrence",
                        "clip_scores",
                        "prompt_added",
                        "prompt_removed",
                        "frame_dropped",
                        "status",
                        "embedder_error",
                    }
                    if e.type == "frame_metrics":
                        seen_metric = True
                        break
                assert seen_metric


class TestCapabilityIsolation:
    def test_spectral_only_no_clip_events(self) -> None:
        """A frame-only session must never emit ClipScores even under load."""
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/v1/events") as ws:
                ws.receive_json()  # session_info
                n_frames = 15
                for i in range(n_frames):
                    ws.send_bytes(_make_jpeg(i))
                # Read exactly n_frames events — color-histogram on random
                # color images produces FrameMetrics per frame with no
                # shot/recurrence/anomaly_alert (no sustained crossings).
                seen_types = set()
                for _ in range(n_frames):
                    msg = ws.receive_json()
                    seen_types.add(msg["type"])
                assert "clip_scores" not in seen_types
                assert "frame_metrics" in seen_types

    def test_full_session_can_emit_clip_events(self) -> None:
        app = create_app(_build_full_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/v1/events") as ws:
                ws.receive_json()  # session_info
                # Add a prompt so clip scores can fire
                ws.send_text(json.dumps({"action": "add_prompt", "text": "a dog"}))
                # PromptAdded arrives after the add_prompt call
                pre_metric_msg = ws.receive_json()
                assert pre_metric_msg["type"] == "prompt_added"

                # Push enough frames to produce two clips (clip_frames=4)
                n_frames = 12
                for i in range(n_frames):
                    ws.send_bytes(_make_jpeg(i))
                # Receive n_frames FrameMetrics (one per frame) plus up to 3
                # ClipScores events fire-and-forget. We just look for clip_scores
                # in the first n_frames+3 events.
                seen_types: set[str] = set()
                for _ in range(n_frames + 3):
                    msg = ws.receive_json()
                    seen_types.add(msg["type"])
                    if "clip_scores" in seen_types:
                        break
                assert "clip_scores" in seen_types


class TestFailureModes:
    def test_warmup_failure_closes_session(self) -> None:
        bad_emb = ImageEmbedder(
            embed_dim=128,
            space_id="bad/wrongshape@128/test",
            embed_fn=lambda frames: np.zeros((len(frames), 64)),  # wrong dim
        )

        def bad_factory() -> Session:
            return Session(frame_embedder=bad_emb)

        app = create_app(bad_factory)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # First message is the SessionInfo-derived "ready" from start()
                first = ws.receive_json()
                assert first["status"] == "ready"
                # Then the warmup error
                err = ws.receive_json()
                assert err["status"] == "error"
                assert "warmup" in err["msg"].lower()


class TestNoRemoteAssets:
    def test_dashboard_loads_no_external_urls(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            html = client.get("/").text
            # No external CDN refs (the only one was Plotly, swapped for /static/)
            assert "https://cdn." not in html
            assert "http://cdn." not in html
            assert "cdn.plot.ly" not in html
            assert "/static/plotly.min.js" in html
