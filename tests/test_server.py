"""Tests for vnvideo.server — app, routes, sinks, dashboard."""
from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.embedders import ColorHistogramEmbedder, DummyEmbedder
from vnvideo.server import create_app
from vnvideo.session import Session


def _build_demo_session() -> Session:
    return Session(
        frame_embedder=ColorHistogramEmbedder.make_image(),
        spectral_config=SpectralConfig(window_frames=5),
    )


def _build_full_session() -> Session:
    return Session(
        frame_embedder=DummyEmbedder.make_image(),
        clip_embedder=DummyEmbedder.make_video(),
        text_embedder=DummyEmbedder.make_text(),
        spectral_config=SpectralConfig(window_frames=5),
        clip_config=__import__("vnvideo.session", fromlist=["ClipConfig"]).ClipConfig(clip_duration_seconds=2.0, clip_stride_seconds=2.0),
        embedder_concurrency=2,
    )


def _make_jpeg() -> bytes:
    img = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class TestHttpRoutes:
    def test_health_returns_200(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_ready"] is True
            assert "version" in data

    def test_api_info_includes_capabilities(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            resp = client.get("/api/v1/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["schema_version"] == "0.1"
            assert data["capabilities"]["spectral_analytics"] is True

    def test_dashboard_served(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "<html" in resp.text or "<!doctype html" in resp.text.lower()

    def test_dashboard_has_local_plotly_not_cdn(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "cdn.plot.ly" not in resp.text
            assert "/static/plotly.min.js" in resp.text

    def test_static_plotly_served(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            resp = client.get("/static/plotly.min.js")
            # Plotly bundle is ~4.5MB
            assert resp.status_code == 200
            assert len(resp.content) > 100_000


class TestLegacyWebSocket:
    def test_handshake_and_session_info(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                msg = ws.receive_json()
                # First message should be a "ready" status (from SessionInfo)
                assert msg["status"] == "ready"

    def test_frame_roundtrip_emits_ok(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                _ = ws.receive_json()  # ready
                ws.send_bytes(_make_jpeg())
                msg = ws.receive_json()
                assert msg["status"] == "ok"
                assert "entropy_norm" in msg
                assert "motion_score" in msg
                assert "anomaly_score" in msg
                assert "buffer_fill" in msg

    def test_bad_jpeg_returns_error(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                _ = ws.receive_json()
                ws.send_bytes(b"not-a-jpeg")
                msg = ws.receive_json()
                assert msg["status"] == "error"
                assert "bad JPEG" in msg["msg"]


class TestNewSchemaWebSocket:
    def test_handshake_emits_session_info_event(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/v1/events") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "session_info"
                assert msg["schema_version"] == "0.1"

    def test_frame_emits_metric_events(self) -> None:
        app = create_app(_build_demo_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/v1/events") as ws:
                _ = ws.receive_json()  # session_info
                ws.send_bytes(_make_jpeg())
                # First few events; we expect at least one frame_metrics
                seen_types = []
                for _ in range(5):
                    try:
                        msg = ws.receive_json()
                        seen_types.append(msg["type"])
                        if msg["type"] == "frame_metrics":
                            assert "entropy_norm" in msg["payload"]
                            break
                    except Exception:
                        break
                assert "frame_metrics" in seen_types


class TestPromptCommand:
    def test_add_prompt_via_ws(self) -> None:
        app = create_app(_build_full_session)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                _ = ws.receive_json()  # ready
                ws.send_text(json.dumps({"action": "add_clip_event", "text": "a dog"}))
                # Eventually we should see a clip_event_added status
                found = False
                for _ in range(5):
                    msg = ws.receive_json()
                    if msg.get("status") == "clip_event_added":
                        found = True
                        assert msg["text"] == "a dog"
                        break
                assert found
