# videospectra examples

Three setup files showing the BYOM (Bring Your Own Model) contract. Each defines a `make_session() -> Session` function; pass the file to `videospectra serve --setup <path>` and the CLI builds the FastAPI app around it.

| File | Embedder | Why use it |
| --- | --- | --- |
| `color_histogram_demo.py` | numpy histograms, 256-d | Install verification. No torch, no GPU. Pure spectral on a meaningless embedder — useful for verifying the pipeline, useless for understanding the scene. |
| `rzenembed_full.py` | RzenEmbed (Qihoo360 Qwen2-VL), 3584-d | The reference. Reproduces the original von-neumann-dashboard demo end-to-end on a GPU node (~15 GB VRAM). |
| `open_clip_frame_pooling.py` | OpenCLIP ViT-L-14, 768-d, frame-pooled clips | Lightweight alternative. Real semantic embeddings, but clips are just averaged frame embeddings — not temporal understanding. |

Run any of them:

```bash
videospectra serve --setup examples/color_histogram_demo.py
# or
videospectra serve --setup examples/rzenembed_full.py --port 8765
```

Then open `http://127.0.0.1:8765/`.

## Writing your own

A setup file is plain Python. The only contract: it must define a function named `make_session` that returns a `Session`. Build any combination of `ImageEmbedder`, `VideoEmbedder`, `TextEmbedder` you want; videospectra handles the rest.

```python
# my_setup.py
from videospectra.embedders import ImageEmbedder
from videospectra.session import Session
from videospectra.analytics.spectral import SpectralConfig

def embed_my_model(frames):
    # frames: list[Frame]; return np.ndarray of shape (len(frames), 512)
    ...

def make_session():
    return Session(
        frame_embedder=ImageEmbedder(
            embed_dim=512,
            space_id="myorg/my-model@512/image-retrieval",
            embed_fn=embed_my_model,
        ),
        spectral_config=SpectralConfig(),
    )
```

See `docs/modularity.md` for the full reference.
