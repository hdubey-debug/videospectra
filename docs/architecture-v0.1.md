# vnvideo v0.1 — Architecture

## Dataflow

```
                       +---------------------+
                       |   FrameSource       |
                       | (WS / video / RTSP) |
                       +----------+----------+
                                  |  Frame
                                  v
              +-------------------+---------------------+
              |                Session                  |
              |  - embedder concurrency = 1 by default  |
              |  - L2-normalizes every embedding        |
              |  - owns SpectralAnalyzer + ClipBuffer   |
              |  - emits events via sinks               |
              +---+-----------+-------------+-----------+
                  |           |             |
        frame emb |   clip emb (async)      | text emb (prompt add)
                  v           v             v
            ImageEmbedder VideoEmbedder TextEmbedder
              (BYOM)        (BYOM)        (BYOM)

              Session emits Event objects -> SinkRunner -> Sink
              (SinkRunner has bounded queue + per-sink worker task)

              Sinks: StdoutSink | JsonlSink | MemorySink | WebSocketSink
```

## Three BYOM contracts

Each is a dataclass wrapping a callable. Required fields are explicit; the framework never auto-fills a `space_id` or guesses a dim.

```python
@dataclass
class ImageEmbedder:
    embed_dim: int
    space_id: str                                      # required, non-empty
    embed_fn: Callable[[list[Frame]], MaybeAwaitable[np.ndarray]]
    name: str = "user"

@dataclass
class VideoEmbedder:
    embed_dim: int
    space_id: str
    embed_fn: Callable[[list[list[Frame]]], MaybeAwaitable[np.ndarray]]  # batched clips
    name: str = "user"

@dataclass
class TextEmbedder:
    embed_dim: int
    space_id: str
    embed_fn: Callable[[list[str]], MaybeAwaitable[np.ndarray]]
    name: str = "user"
```

Notes:

- The callable may be sync or async. The Session dispatches via `inspect.iscoroutinefunction` + `inspect.isawaitable`.
- Outputs are validated for shape `(N, embed_dim)`, finite values, and dtype castable to float64. The framework then applies L2 normalization unconditionally.
- `space_id` is opaque and required. It is checked for cross-space mismatches (`clip_embedder.space_id == text_embedder.space_id` is a constructor invariant).
- There is no `normalized` flag on the wrapper. The framework normalizes regardless of what the user returns.
- There is no `metadata` escape hatch on the wrapper or on `Frame`.

## Session API

```python
class Session:
    def __init__(
        self,
        *,
        frame_embedder: ImageEmbedder | None = None,
        clip_embedder: VideoEmbedder | None = None,
        text_embedder: TextEmbedder | None = None,
        sinks: Sequence[Sink] = (),
        spectral_config: SpectralConfig | None = None,
        clip_config: ClipConfig | None = None,
        source_fps: float = 2.0,
        session_id: str | None = None,
        embedder_concurrency: int = 1,
        sink_max_queue: int = 100,
        sink_overflow: Literal["drop_oldest", "drop_newest", "block"] = "drop_oldest",
    ) -> None: ...

    async def start(self) -> None: ...
    async def aclose(self) -> None: ...

    async def process_frame(self, frame: Frame) -> None: ...

    async def add_prompt(self, text: str) -> str: ...
    async def remove_prompt(self, prompt_id: str) -> None: ...
    def list_prompts(self) -> list[dict]: ...

    async def add_sink(self, sink: Sink) -> None: ...
    async def remove_sink(self, sink: Sink) -> None: ...

    def capabilities(self) -> Capabilities: ...
    async def warmup(self) -> None: ...
```

```python
@dataclass(frozen=True)
class ClipConfig:
    clip_duration_seconds: float = 4.0   # length of each clip in seconds
    clip_stride_seconds: float = 4.0     # gap between successive clip starts
    max_events: int = 10
```

Key invariants enforced at construction:

- At least one of `{frame_embedder, clip_embedder}` must be set
- If `clip_embedder` is set, `text_embedder` must also be set, and their `space_id` and `embed_dim` must match
- If `clip_embedder` is set, `clip_config` is required
- If `frame_embedder` is set and `spectral_config` is set, the spectral pipeline activates
- `source_fps` must be > 0

`process_frame` returns `None`. All events leave via sinks. A slow clip embed never blocks the frame loop — clip work runs in a fire-and-forget task.

### Frame rate

`source_fps` is the rate at which the caller promises the upstream frame source will deliver frames. The Session does **not** sample or decimate — driving the source at the promised rate is the source's responsibility (the browser dashboard, video reader, or RTSP capture). `source_fps` is used internally to:

- Resolve `ClipConfig.clip_duration_seconds` / `clip_stride_seconds` into integer frame counts: `clip_frames = max(1, round(clip_duration_seconds * source_fps))`. The Session logs a warning if rounding shifts the requested seconds by more than 5%.
- Report seconds-accurate timestamps in `SessionInfo.config`.

`SpectralConfig.window_frames` stays frame-count based — eigendecomposition is fundamentally counted in samples, not seconds. Convert manually if you need a fixed time window: `window_frames = round(seconds * source_fps)`.

## Frame

```python
@dataclass(frozen=True)
class Frame:
    image: PIL.Image.Image
    frame_id: int
    source_id: str
    monotonic_ts: float        # perf_counter()  — latency math only
    wall_ts: float             # time.time()     — logging / display only
    width: int
    height: int

    @classmethod
    def from_jpeg(cls, data: bytes, *, source_id: str, frame_id: int) -> Frame: ...
    @classmethod
    def from_pil(cls, image: PIL.Image.Image, *, source_id: str, frame_id: int) -> Frame: ...
```

`frozen=True` — Frame is immutable and hashable. No metadata escape hatch.

## Sinks

```python
@runtime_checkable
class Sink(Protocol):
    name: str
    async def emit(self, event: Event) -> None: ...
    async def aclose(self) -> None: ...
```

`SinkRunner` (internal): one worker task per sink, bounded `asyncio.Queue`, overflow policy `drop_oldest | drop_newest | block`. A sink that raises `failure_threshold` times in a row is disabled and logged. The Session never blocks on `emit` directly — only on `enqueue`, which is non-blocking under drop policies.

Built-in sinks:

| Sink | Purpose |
| --- | --- |
| `StdoutSink` | One JSON line per event to stdout |
| `JsonlSink(path)` | Append-only newline-delimited JSON file |
| `MemorySink` | `asyncio.Queue`-backed; iterable; the canonical SDK consumption pattern |
| `WebSocketSink(ws)` | Lives in `server/`; serializes via `model_dump(mode='json')` |

## Per-frame emission ordering

For each frame, the Session emits events in this order:

1. `ShotBoundary` — if the rising-edge anomaly detector fired
2. `AnomalyAlert` — if a sustained run crossed the threshold this frame
3. `FrameMetrics` — always last per frame

`FrameMetrics` is the per-frame "flush" signal that aggregators (e.g., the bundled legacy dashboard sink) rely on to render one combined UI update. The conditional events (`ShotBoundary`, `AnomalyAlert`) belong to the same frame's processing and arrive *before* the `FrameMetrics` for that frame_id.

`ClipScores` is asynchronous — it fires when a clip-level embedder finishes a clip (every ~8 frames at 2 FPS, in a fire-and-forget task). It carries the `frame_id` of the last frame in the clip but is not bound to that frame's per-frame ordering.

## Event envelope

Every event is a Pydantic model that serializes as:

```json
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
```

Discriminated by `type`. Subclasses include `frame_metrics`, `shot_boundary`, `anomaly_alert`, `clip_scores`, `prompt_added`, `prompt_removed`, `frame_dropped`, `status`, `embedder_error`, `session_info`. Discriminated parsing is via `Event = Annotated[Union[...], Field(discriminator='type')]`.

## Embedder concurrency

`embedder_concurrency=1` is the default. The Session holds a single `asyncio.Semaphore(embedder_concurrency)` that wraps every embedder call (frame, clip, text). This serializes calls into single-threaded model backends and avoids CUDA OOM under bursty traffic.

Sync embed functions are dispatched via `loop.run_in_executor(ThreadPoolExecutor(max_workers=max(1, embedder_concurrency)))`. Async embed functions are awaited directly. A returned awaitable from a sync function is also awaited (covers wrappers that return a coroutine from a synchronous Python function — e.g., `functools.partial` of an async fn).

## Spectral analytics

`vnvideo/analytics/spectral.py` ports the math from `server.py:177-306` byte-equivalent. The `SpectralAnalyzer` class owns the sliding window, eigendecomposition, motion / anomaly / shot state. It is GPU-free, pure numpy.

```python
@dataclass
class SpectralConfig:
    window_frames: int = 30
    top_k_eigen: int = 30
    top_k_vec: int = 3
    anomaly_threshold: float = 0.5
    ema_alpha: float = 0.3

@dataclass
class SpectralUpdate:
    entropy_norm: float
    motion_score: float
    anomaly_score: float
    is_anomaly: bool
    is_shot_boundary: bool
    shot_count: int
    buffer_fill: int
    infer_ms_spectral: float

class SpectralAnalyzer:
    def __init__(self, config: SpectralConfig) -> None: ...
    def update(self, embedding: np.ndarray) -> SpectralUpdate: ...
    def reset(self) -> None: ...
```

Shot boundary uses rising-edge detection on `is_anomaly` (not the raw score). All state lives in the analyzer; Session does not reach inside.

## Hard invariants

The same 13 from `requirements-v0.1.md`, restated as one-line rules a reviewer can verify mechanically:

1. `vnvideo/session.py` does not import `socket`, `fastapi`, `uvicorn`, or `open()` files
2. `Session.process_frame` returns `None`
3. `Session.add_sink` wraps the sink in `SinkRunner` with `sink_max_queue` and `sink_overflow`
4. `Session.__init__(embedder_concurrency=1)` is the default
5. `Session` calls `l2_normalize` on every embedding the embedder returns, unconditionally; no `normalized` field exists on `ImageEmbedder` / `VideoEmbedder` / `TextEmbedder`
6. `ImageEmbedder.__post_init__` raises on empty `space_id`; `Session.__init__` raises on `clip_embedder.space_id != text_embedder.space_id`
7. `vnvideo/analytics/` contains exactly one module: `spectral.py`
8. `vnvideo/events.py` defines all events and does not `import fastapi`
9. `vnvideo/cli.py` defaults `--host` to `127.0.0.1` and emits a warning if the user passes `0.0.0.0`
10. `grep "https://" vnvideo/server/static/dashboard.html` returns nothing
11. To add a new ImageEmbedder backend, the user creates a setup file outside `vnvideo/` — no edits to `vnvideo/` source
12. Adding a new role (e.g., audio embedder) requires editing `vnvideo/embedders.py`, `vnvideo/session.py`, `vnvideo/events.py`
13. Session is FPS-aware via `source_fps`; `ClipConfig` is seconds-based, resolved to integer frames at construction
