# vnvideo v0.1 — Modularity Guide

vnvideo is modular along five axes. Three are externally pluggable (you write code outside `vnvideo/`); two are internally modular (changes require editing core).

| Axis | External? | How |
| --- | --- | --- |
| Embedders | Yes | Wrap any callable in `ImageEmbedder` / `VideoEmbedder` / `TextEmbedder` |
| Sources | Yes | Implement the `FrameSource` Protocol |
| Sinks | Yes | Implement the `Sink` Protocol |
| Analytics | No | Edit `vnvideo/analytics/` and `vnvideo/session.py` |
| Event types | No | Edit `vnvideo/events.py` and consumers |

## How to add an embedder backend (external)

Write a setup file. The CLI loads it as a Python module and calls `make_session()`.

```python
# my_setup.py
import numpy as np
from PIL import Image
from vnvideo import Session, ImageEmbedder
from vnvideo.analytics import SpectralConfig

def embed_my_model(frames):
    # frames: list[Frame]; return np.ndarray of shape (len(frames), 512)
    arr = np.stack([np.asarray(f.image.resize((224, 224))).ravel() for f in frames])
    return (arr / 255.0)[:, :512].astype(np.float32)

def make_session() -> Session:
    return Session(
        frame_embedder=ImageEmbedder(
            embed_dim=512,
            space_id="myorg/my-model@512/image-retrieval",
            embed_fn=embed_my_model,
            name="my-model",
        ),
        spectral_config=SpectralConfig(),
    )
```

Then: `vnvideo serve --setup my_setup.py`.

Notes:

- `space_id` is required and must be non-empty. Pick a string unique to your model + dimension + task; vnvideo only checks it for equality across the clip/text pair.
- vnvideo always re-applies L2 normalization to your output. You can pre-normalize or not — the math is invariant.
- Your `embed_fn` can be sync or async. If sync, it runs in a thread pool. If async, it is awaited.

## How to add a frame source (external)

```python
from vnvideo import Frame, Session
from vnvideo.sources import FrameSource, run_source

class MyVideoSource:
    source_id = "my-video"

    async def frames(self):
        for i in range(100):
            pil = ... # produce a PIL.Image
            yield Frame.from_pil(pil, source_id=self.source_id, frame_id=i)

    async def aclose(self):
        pass

async def main(session: Session):
    await run_source(MyVideoSource(), session)
```

`FrameSource` is structurally typed (`runtime_checkable` Protocol). Any object with the right attributes / methods works.

## How to add a sink (external)

```python
from vnvideo.sinks import Sink
from vnvideo.events import Event

class MySink:
    name = "my-sink"

    async def emit(self, event: Event) -> None:
        # do something — POST to a webhook, write to a DB, ...
        ...

    async def aclose(self) -> None:
        ...

session = Session(..., sinks=[MySink()])
```

The Session wraps your sink in a `SinkRunner` automatically. A failing sink is disabled after 5 consecutive errors; a slow sink does not block the frame loop (events queue up to `sink_max_queue` and the oldest ones drop by default).

## How to add a setup file with the dashboard

Setup files are pure Python — there is no DSL. The contract is just: define a `make_session() -> Session` function. The CLI imports the file via `importlib.util.spec_from_file_location`, looks up `make_session`, and hands the returned `Session` to the FastAPI app.

The reference dashboard at `http://127.0.0.1:8765` reads events from the WebSocket your session emits via the built-in `WebSocketSink`. The dashboard does not care which model you ran — it consumes the versioned event schema.
