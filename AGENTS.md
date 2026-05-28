# videospectra — guide for AI coding agents

You are operating inside `videospectra`, a Python package for spectral video understanding (von Neumann entropy + motion / anomaly / shot detection + clip scoring). Read this file in full before making changes. If the user asks you to "explain this project," lead with the architecture tree in §2 and keep it compact; do not paraphrase the marketing.

## 0. Project one-liner

A source-agnostic, BYOM (bring-your-own-model) video analytics library: image / video / text embedder wrappers feed a Session that emits Pydantic events through pluggable sinks (stdout, JSONL, memory, WebSocket). Pure-numpy spectral analytics; FastAPI + bundled dashboard ships as an optional `[server]` extra.

## 1. Repository layout

```
videospectra/                       Python package
├── _internal/normalize.py     L2 normalize (called unconditionally by Session)
├── types.py                   Frame, Capabilities
├── embedders.py               ImageEmbedder / VideoEmbedder / TextEmbedder + helpers
├── events.py                  Pydantic event schemas (discriminated union)
├── sinks.py                   Sink Protocol + SinkRunner backpressure
├── analytics/spectral.py      SpectralAnalyzer (pure numpy, < 5ms/frame CPU)
├── session.py                 Session orchestrator; ClipConfig (seconds-based)
├── sources.py                 FrameSource Protocol + run_source helper
├── server/
│   ├── app.py                 create_app FastAPI factory
│   ├── routes.py              HTTP + WebSocket routes
│   ├── sinks.py               WebSocketSink + LegacyDashboardWebSocketSink
│   └── static/dashboard.html  bundled UI (Plotly served locally — no CDN)
└── cli.py                     `videospectra demo` + `videospectra serve --setup`

tests/                         pytest, GPU-free
├── fixtures/legacy_compute_entropy.py    FROZEN — parity source of truth
└── integration/test_parity_with_legacy.py

examples/                      three working setup files
├── color_histogram_demo.py    CPU only
├── open_clip_frame_pooling.py modest GPU
└── rzenembed_full.py          ~15 GB VRAM, full demo

docs/
├── architecture-v0.1.md       dataflow, contracts, invariants
├── requirements-v0.1.md       in/out of scope, 13 hard invariants
└── modularity.md              extension points
```

## 2. Architecture tree

```
                                      +---------------------+
                                      |   FrameSource       |
                                      | (your WS/video/RTSP)|
                                      +----------+----------+
                                                 |  Frame
                                                 v
   ImageEmbedder   --(per frame)-->   +----------+----------+
   VideoEmbedder   --(per clip)-->    |       Session       |   --(events)-->   Sinks
   TextEmbedder    --(per prompt)-->  |  source_fps         |                   ├── StdoutSink
                                      |  embedder_concurrency|                  ├── JsonlSink
                                      |  spectral_config    |                   ├── MemorySink
                                      |  clip_config (sec)  |                   └── WebSocketSink
                                      +----+--------+-------+                       (LegacyDashboardWebSocketSink for /ws)
                                           |        |
                                  frame_emb|        |clip_emb (fire-and-forget)
                                           v        v
                                +----------+----+   +------+------+
                                |SpectralAnalyzer|   |  ClipScorer  |
                                |  entropy       |   |  cosine sim  |
                                |  motion        |   |  vs prompts  |
                                |  anomaly       |   +-------------+
                                |  shot boundary |
                                +----------------+
```

Per-frame emission order, enforced in `Session._process_frame_inner`:

1. `ShotBoundary` (if rising-edge anomaly fired)
2. `AnomalyAlert` (if sustained run crossed threshold)
3. `FrameMetrics` (always last — the per-frame "flush" signal aggregating sinks rely on)

`ClipScores` is asynchronous (one per clip stride). `PromptAdded`/`PromptRemoved`/`FrameDropped`/`SessionInfo`/`Status`/`EmbedderError` arrive whenever they happen.

## 3. How to add things (recipes)

### New ImageEmbedder backend (e.g. CLIP, DINO, your own model)

```python
# my_setup.py  — outside videospectra/, no edits to videospectra source
from videospectra.embedders import ImageEmbedder
from videospectra.session import Session
from videospectra.analytics.spectral import SpectralConfig

def embed_fn(frames):  # list[Frame] -> np.ndarray (N, embed_dim)
    return my_model.encode([f.image for f in frames])

def make_session():
    return Session(
        frame_embedder=ImageEmbedder(
            embed_dim=512, space_id="myorg/mymodel@512/image", embed_fn=embed_fn,
        ),
        spectral_config=SpectralConfig(),
        source_fps=2.0,
    )
```

Then: `videospectra serve --setup my_setup.py`.

### New Sink

Implement the `Sink` Protocol (`name: str`, `async def emit(event)`, `async def aclose()`), then `await session.add_sink(my_sink)`. SinkRunner wraps it with a bounded queue and drop policy. Don't block in `emit` — backpressure is the framework's job.

### New FrameSource

Implement the `FrameSource` Protocol in `videospectra/sources.py`: an async iterator of `Frame`. Drive it via `run_source(source, session)`. Source is responsible for hitting the promised `source_fps` — the Session does not sample or decimate.

## 4. Hard invariants (do not violate)

1. `videospectra/session.py` does not import `socket`, `fastapi`, `uvicorn`, or open files
2. `Session.process_frame` returns `None`; all events leave via sinks
3. `Session.add_sink` wraps the sink in `SinkRunner` with `sink_max_queue` and `sink_overflow`
4. `Session.__init__(embedder_concurrency=1)` is the default
5. `Session` calls `l2_normalize` on every embedding unconditionally; no `normalized` flag exists on any BYOM wrapper
6. `ImageEmbedder.__post_init__` raises on empty `space_id`; `Session.__init__` raises on `clip_embedder.space_id != text_embedder.space_id`
7. `videospectra/analytics/` contains exactly one module: `spectral.py`
8. `videospectra/events.py` defines all events and does not `import fastapi`
9. `videospectra/cli.py` defaults `--host` to `127.0.0.1`; warns on `0.0.0.0`
10. `grep "https://" videospectra/server/static/dashboard.html` returns nothing (no CDN assets)
11. Adding a new implementation of an existing role: zero changes to `videospectra/`
12. Adding a new role or analytic IS a core change — don't pretend otherwise
13. Session is FPS-aware via `source_fps`; `ClipConfig` is seconds-based, resolved to integer frames at construction

## 5. Pitfalls

- **Pydantic v2 + Python 3.9**: Pydantic resolves annotation strings at runtime via `get_type_hints`; PEP 604 `X | Y` fails to eval. `videospectra/events.py` uses `typing.Optional` / `typing.Union` even though ruff prefers `X | Y`. The per-file ruff ignore is in `pyproject.toml` — leave it alone.
- **Parity gate**: `tests/fixtures/legacy_compute_entropy.py` is FROZEN. It's a verbatim copy of `compute_entropy` from `von-neumann-dashboard/server.py:177-306` and `TestLegacyFixtureFreshness` will fail if it drifts from the live source. The parity test asserts new spectral fields match within 1e-9. Do not edit the fixture; if the upstream file changes, regenerate by copy-paste and re-pin.
- **`effective_rank` is gone** from the public API but still computed in the frozen fixture — that's intentional. The parity assertion for `effective_rank` was removed; don't add it back.
- **`Recurrence` is gone** entirely (event, payload, config knobs, spectral state). Don't reintroduce.
- **Per-frame emission order**: `ShotBoundary` / `AnomalyAlert` FIRST, `FrameMetrics` LAST. The legacy dashboard sink buffers partials by `frame_id` and flushes on FrameMetrics. Reordering breaks the UI.
- **mypy must be run explicitly** — `ruff check` does not type-check. Run both before declaring done.
- **COVERAGE_FILE on shared FS**: pytest-cov uses SQLite. On the shared filesystem you'll get lock errors unless you set `COVERAGE_FILE=/tmp/.coverage.videospectra`.
- **GPU is optional**: tests must stay GPU-free. Use `DummyEmbedder` / `ColorHistogramEmbedder` in tests.
- **`Session._resolved_clip` is private** (leading underscore) but the dashboard route reads it. That's OK — `routes.py` lives in the same package — but don't read it from user-facing code; the seconds-based `ClipConfig` is the contract.

## 6. Common commands

```bash
# Always activate the venv first
source /mmfs2/home/jacks.local/hdubey/scratch/02-video-embeddings/rzenembed/rzen_venv/bin/activate

cd /mmfs1/scratch/jacks.local/hdubey/02-video-embeddings/rzenembed/videospectra

# Full test suite (155 tests, < 10 s, ~90% coverage)
COVERAGE_FILE=/tmp/.coverage.videospectra pytest tests/ --cov=videospectra --cov-report=term-missing

# Lint + type
ruff check videospectra/ tests/ examples/
mypy videospectra/

# Run the no-GPU demo
videospectra demo                 # http://127.0.0.1:8765

# Run a custom setup
videospectra serve --setup examples/rzenembed_full.py --port 8765

# Build wheel/sdist
python -m build
```

## 7. Deep dives

- `docs/architecture-v0.1.md` — dataflow, BYOM wrappers, Session API, per-frame emission ordering, frame rate, hard invariants
- `docs/requirements-v0.1.md` — in/out of scope, success criteria, the 13 hard invariants
- `docs/modularity.md` — adding embedder backends, sinks, sources
- `examples/` — three real setup files; copy and adapt
