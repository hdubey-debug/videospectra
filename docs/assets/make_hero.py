"""Regenerate the README hero animation (docs/assets/hero.gif).

Runs the REAL videospectra pipeline (Session + ColorHistogramEmbedder +
spectral analytics) over 50 synthetic frames with a scene change at
frame 25, then animates the actual entropy / motion / anomaly traces
beside the input frame and marks the detected shot boundary. Every
number plotted is computed by the library — there is no mockup.

Usage (from a checkout with videospectra installed)::

    pip install -e ".[dev]" matplotlib
    python docs/assets/make_hero.py

Outputs ``hero.gif`` next to this script.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from videospectra.analytics.spectral import SpectralConfig
from videospectra.embedders import ColorHistogramEmbedder
from videospectra.events import FrameMetrics, ShotBoundary
from videospectra.session import Session
from videospectra.sinks import MemorySink
from videospectra.types import Frame

OUT = Path(__file__).resolve().parent / "hero.gif"


def make_synthetic_frames(n=50, size=64, scene_change_at=25, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        scene = 0 if i < scene_change_at else 1
        t = (i if scene == 0 else (i - scene_change_at)) / max(1, scene_change_at)
        if scene == 0:
            r, g, b = int(40 + 20 * t), int(60 + 20 * t), int(200 - 20 * t)
        else:
            r, g, b = int(220 - 20 * t), int(60 + 20 * t), int(40 + 20 * t)
        base = np.zeros((size, size, 3), dtype=np.uint8)
        base[..., 0], base[..., 1], base[..., 2] = r, g, b
        noise = rng.integers(-10, 11, size=(size, size, 3), dtype=np.int16)
        arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


async def _run(frames):
    sink = MemorySink()
    session = Session(
        frame_embedder=ColorHistogramEmbedder.make_image(),
        spectral_config=SpectralConfig(window_frames=10),
        sinks=[sink],
        source_fps=2.0,
    )
    await session.start()
    for i, img in enumerate(frames):
        await session.process_frame(Frame.from_pil(img, source_id="synth", frame_id=i))
    await session.aclose()
    events = []
    async for event in sink:
        events.append(event)
    return events


def main() -> None:
    plt.switch_backend("Agg")  # headless: render to file, no display needed
    frames = make_synthetic_frames()
    events = asyncio.run(_run(frames))

    fids, ent, mot, ano, shots = [], [], [], [], []
    for e in events:
        if isinstance(e, FrameMetrics):
            fids.append(e.frame_id)
            ent.append(e.payload.entropy_norm)
            mot.append(e.payload.motion_score)
            ano.append(e.payload.anomaly_score)
        elif isinstance(e, ShotBoundary):
            shots.append(e.frame_id)
    fids = np.array(fids)
    ent, mot, ano = np.array(ent), np.array(mot), np.array(ano)

    bg, fg, mute, grid = "#0d1117", "#c9d1d9", "#8b949e", "#21262d"
    c_ent, c_mot, c_ano = "#58a6ff", "#f0b72f", "#ff7b72"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "xtick.color": mute,
        "ytick.color": mute, "axes.edgecolor": grid, "font.size": 11,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(8.4, 2.7), dpi=110, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.0])
    ax_img = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("input frame", fontsize=9, color=mute)
    im = ax_img.imshow(np.asarray(frames[0]), interpolation="nearest")
    for sp in ax_img.spines.values():
        sp.set_edgecolor(grid)

    (l_ent,) = ax.plot([], [], color=c_ent, lw=2.0, label="entropy")
    (l_mot,) = ax.plot([], [], color=c_mot, lw=2.0, label="motion")
    (l_ano,) = ax.plot([], [], color=c_ano, lw=2.0, label="anomaly")
    ax.set_xlim(0, len(fids) - 1)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("frame", fontsize=9)
    ax.set_title("videospectra  ·  spectral signals computed live",
                 fontsize=12, color=fg, loc="left")
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor=grid,
              labelcolor=fg, ncol=3, fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.18, color=grid)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid)

    drawn = set()

    def update(t):
        im.set_data(np.asarray(frames[t]))
        k = t + 1
        l_ent.set_data(fids[:k], ent[:k])
        l_mot.set_data(fids[:k], mot[:k])
        l_ano.set_data(fids[:k], ano[:k])
        for sf in shots:
            if sf <= fids[t] and sf not in drawn:
                ax.axvline(sf, color=c_ano, ls="--", alpha=0.6, lw=1.3)
                ax.annotate("shot\nboundary", xy=(sf, 1.0), xytext=(sf + 0.6, 0.72),
                            color=c_ano, fontsize=8.5, ha="left", va="top")
                drawn.add(sf)
        return l_ent, l_mot, l_ano, im

    anim = FuncAnimation(fig, update, frames=len(frames), interval=110, blit=False)
    anim.save(str(OUT), writer=PillowWriter(fps=11))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); shots at {shots}")


if __name__ == "__main__":
    main()
