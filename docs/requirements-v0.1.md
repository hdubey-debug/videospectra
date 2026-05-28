# videospectra v0.1 — Requirements

## Why this exists

The von-neumann-dashboard demo proved that a small spectral pipeline (von Neumann entropy of a sliding embedding window, plus motion / anomaly / shot signals) gives a useful real-time read on video content. v0.1 turns that working demo into a publishable Python package so that:

- researchers can import `SpectralAnalyzer` and feed in any embedding sequence
- tinkerers can `pip install`, hand videospectra a small "setup file" that loads their model, and get a working web dashboard at `127.0.0.1:8765`
- integrators can attach to the WebSocket+REST API with a versioned schema and consume events without ever loading a model in-process

## Personas

| Persona | Wants | Gives up |
| --- | --- | --- |
| Researcher | Pure ndarray analytics, no torch in base, drop-in `SpectralAnalyzer` | Server, UI |
| Tinkerer | `videospectra demo` runs locally with zero-config; can swap in own model later | Multi-camera, hardened auth |
| Integrator | Documented WebSocket+REST API, stable event schemas, JSONL persistence | Hand-holding |

## In scope (v0.1)

1. Single Python package `videospectra` with three BYOM contracts (`ImageEmbedder`, `VideoEmbedder`, `TextEmbedder`)
2. Spectral analytics module ported from `server.py:177-306` with byte-equivalent metrics
3. Sink-based event pipeline with backpressure (`SinkRunner` + bounded queues)
4. Source-agnostic `Session` orchestrator with embedder-concurrency safety by default
5. Versioned Pydantic event schemas in core (no FastAPI dep)
6. FastAPI server adapter — thin wrapper over `Session`
7. Bundled reference dashboard (content unchanged from demo; only the Plotly CDN URL is replaced with a local asset)
8. `videospectra serve --setup <file>` and `videospectra demo` CLI entry points
9. Three example setup files: `rzenembed_full.py`, `open_clip_frame_pooling.py`, `color_histogram_demo.py`
10. JSONL persistence as a built-in sink
11. Apache 2.0 license
12. Unit + integration tests; documented contract

## Out of scope (deferred)

- `videospectra analyze <video>` batch CLI
- Local webcam frame source (`opencv-python`)
- RTSP source
- Multi-session / multi-camera
- UI redesign
- Adaptive sampling
- API-backend examples (OpenAI, Voyage, Gemini, Vertex)
- Docker images
- TypeScript client library
- Plugin discovery / registration
- Authentication
- Jetson hardware validation (architecture supports it; physical test deferred)

## Success criteria

- `pip install -e ./videospectra[server,dev]` succeeds without a GPU
- `pytest` is GPU-free (uses `DummyEmbedder` / `ColorHistogramEmbedder`)
- `videospectra demo` opens a working dashboard at `http://127.0.0.1:8765`
- `videospectra serve --setup examples/rzenembed_full.py` reproduces the original demo end-to-end on a GPU node
- All 13 architecture invariants enforced (see `architecture-v0.1.md`)

## The 13 hard invariants

1. `Session` owns processing state only — never opens sockets, never reads files
2. Events flow exclusively via sinks. `process_frame(frame) -> None`
3. Sinks are wrapped in `SinkRunner` with bounded queues; slow sinks never block the frame loop
4. Embedder calls are serialized by default (`embedder_concurrency=1`)
5. Embeddings are L2-normalized at the Session boundary, unconditionally — no `normalized` flag
6. `space_id` is required, non-empty, opaque. `clip.space_id == text.space_id` is enforced
7. Spectral analytics live in one cohesive module (`analytics/spectral.py`)
8. Event models live in core (`videospectra/events.py`), not under `server/`
9. Server binds `127.0.0.1` by default; `--host 0.0.0.0` requires explicit opt-in
10. Reference UI loads zero remote assets — Plotly is bundled in `videospectra/server/static/`
11. Adding a new **implementation** of an existing embedder/source/sink role requires zero changes to `videospectra/`
12. Adding a new **role** or **analytic** is a core change — we don't pretend otherwise
13. Session is FPS-aware via `source_fps`; clip windowing and time math derive from it. `ClipConfig` is seconds-based, resolved to integer frames at construction.
