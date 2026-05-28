# vnvideo

Spectral video understanding as a library. Drop in your own image / video / text embedder, push frames in, get entropy / motion / anomaly / shot-boundary / clip-similarity events out.

```
                    +-----------------------+
   frames (PIL) --> |       Session         | --> events (Pydantic)
                    |  spectral + clip-score |
                    +-----------------------+
                       ^                ^
                       |                |
                  ImageEmbedder     sinks: stdout / jsonl / memory / websocket
                  VideoEmbedder
                  TextEmbedder
```

Status: **v0.1 alpha.** API is not yet frozen; see `docs/requirements-v0.1.md` for the 13 hard invariants we won't break inside v0.1.

## What this gives you

```
vnvideo
├── Bring-Your-Own-Model embedders
│   ├── ImageEmbedder  — per-frame features
│   ├── VideoEmbedder  — clip-level features
│   └── TextEmbedder   — prompts for clip scoring
├── Session — orchestrator (process_frame -> events)
│   ├── L2-normalizes every embedding
│   ├── serializes embedder calls (default concurrency=1, GPU-safe)
│   ├── source_fps + ClipConfig in seconds (clip_duration_seconds / clip_stride_seconds)
│   └── fire-and-forget clip embedding (slow clips never block frames)
├── Spectral analytics (pure numpy, < 5 ms / frame on CPU)
│   ├── von Neumann entropy of the sliding window
│   ├── motion score (spectral leverage)
│   ├── anomaly score (subspace-fit residual)
│   └── shot boundary (rising edge on anomaly)
├── Events (Pydantic, discriminated union, versioned schema)
│   ├── FrameMetrics / ShotBoundary / AnomalyAlert / ClipScores
│   ├── PromptAdded / PromptRemoved / FrameDropped
│   └── SessionInfo / Status / EmbedderError
├── Sinks (with backpressure: SinkRunner + bounded queue)
│   ├── StdoutSink / JsonlSink / MemorySink
│   └── WebSocketSink (canonical) + LegacyDashboardWebSocketSink
├── FastAPI server adapter  — thin wrapper over Session
│   ├── /ws/v1/events  (canonical event stream)
│   ├── /ws            (bundled dashboard, aggregated format)
│   ├── /api/v1/info, /health, prompt CRUD
│   └── bundled dashboard (Plotly served locally, zero CDN)
└── CLI
    ├── vnvideo demo                  — ColorHistogramEmbedder, no GPU
    └── vnvideo serve --setup <file>  — load your make_session() factory
```

## What this does NOT give you (yet)

```
out-of-scope-in-v0.1
├── Built-in model integrations    — you wrap them (examples/ shows three)
├── Webcam / RTSP frame sources    — bring your own (FrameSource Protocol)
├── Batch CLI (vnvideo analyze)    — coming in v0.2
├── Multi-camera / multi-session   — one Session per process
├── Authentication                 — binds 127.0.0.1 by default
├── Adaptive sampling              — caller drives the FPS
├── TypeScript client              — connect to /ws/v1/events directly
└── UI redesign                    — the bundled dashboard is the legacy demo
```

## Install

```bash
pip install -e .[server,dev]
```

`server` extras pull in FastAPI + uvicorn; `dev` extras pull in pytest / ruff / mypy. The core library has no torch dependency.

## 30-second hello

```bash
vnvideo demo                 # no GPU, ColorHistogramEmbedder + spectral
# open http://127.0.0.1:8765, allow webcam, watch entropy/motion/shot light up
```

## Bring your own model

```python
import numpy as np
from PIL import Image
from vnvideo.embedders import ImageEmbedder
from vnvideo.session import Session, ClipConfig
from vnvideo.analytics.spectral import SpectralConfig
from vnvideo.sinks import MemorySink

def embed_frames(frames):
    # frames: list[Frame]  — frames[i].image is a PIL.Image
    # return shape: (N, embed_dim), any float dtype, finite
    arr = np.stack([np.asarray(f.image, dtype=np.float32).mean(axis=(0, 1)) for f in frames])
    return arr  # toy 3-d "embedding"

embedder = ImageEmbedder(
    embed_dim=3,
    space_id="example/mean-rgb@3/demo",
    embed_fn=embed_frames,
)

sink = MemorySink()
session = Session(
    frame_embedder=embedder,
    spectral_config=SpectralConfig(window_frames=30),
    sinks=[sink],
    source_fps=2.0,
)

# Push frames in (async):
#   await session.start()
#   await session.process_frame(Frame.from_pil(img, source_id="cam0", frame_id=i))
#   ...consume events from `sink`
```

Want a clip-level model too? Add `clip_embedder` + `text_embedder` (same `space_id`, same `embed_dim`) and `clip_config=ClipConfig(clip_duration_seconds=4.0, clip_stride_seconds=4.0)`. See `examples/rzenembed_full.py` and `examples/open_clip_frame_pooling.py` for two real configurations.

## Going deeper

- `docs/architecture-v0.1.md` — full contract: dataflow, BYOM wrappers, Session API, per-frame emission ordering, frame rate, hard invariants
- `docs/requirements-v0.1.md` — in/out of scope and the 13 hard invariants
- `docs/modularity.md` — extension points: adding new embedder backends, sinks, sources
- `examples/` — three working setup files (color histogram, open_clip, rzenembed)
- `CLAUDE.md` — guide for AI coding agents (Claude Code, Codex). Point your agent at this file before asking it to operate on the repo.

Licensed under Apache 2.0.
