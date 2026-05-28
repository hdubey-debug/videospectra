"""``videospectra`` CLI entrypoint.

Two subcommands in v0.1:

- ``videospectra serve --setup <path>`` — import the user's setup file,
  call ``make_session()`` to build a session, and run the FastAPI app
  on the bundled dashboard.
- ``videospectra demo`` — run the same server with the built-in
  ColorHistogramEmbedder so users can verify a local install without
  any model. Pure numpy.

Defaults are safe: ``--host 127.0.0.1`` (loopback only), no browser
auto-open, no torch import. ``--host 0.0.0.0`` prints a warning since
it exposes the dashboard on all interfaces.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Callable

from videospectra import __version__

logger = logging.getLogger("videospectra.cli")


def _build_demo_session_factory() -> Callable:
    """Build a make_session() that returns a ColorHistogramEmbedder-only Session."""
    # Imported lazily so `videospectra --version` doesn't pull in fastapi / etc.
    from videospectra.analytics.spectral import SpectralConfig
    from videospectra.embedders import ColorHistogramEmbedder
    from videospectra.session import Session

    def make_session():
        return Session(
            frame_embedder=ColorHistogramEmbedder.make_image(),
            spectral_config=SpectralConfig(window_frames=30),
            source_fps=2.0,
        )

    return make_session


def _load_setup_module(setup_path: Path) -> Callable:
    """Import the setup file and return its ``make_session`` function."""
    if not setup_path.exists():
        print(f"videospectra: setup file not found: {setup_path}", file=sys.stderr)
        sys.exit(1)
    if not setup_path.is_file():
        print(f"videospectra: setup path is not a file: {setup_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("videospectra_user_setup", setup_path)
    if spec is None or spec.loader is None:
        print(f"videospectra: failed to load setup file: {setup_path}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["videospectra_user_setup"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"videospectra: setup file raised on import: {exc}", file=sys.stderr)
        sys.exit(1)

    make_session = getattr(module, "make_session", None)
    if make_session is None or not callable(make_session):
        print(
            f"videospectra: setup file {setup_path} must define a callable `make_session() -> Session`",
            file=sys.stderr,
        )
        sys.exit(1)
    return make_session


def _launch_server(
    session_factory: Callable,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Build the FastAPI app and run it under uvicorn."""
    # Imports happen here so `videospectra --version` is dependency-light.
    import uvicorn

    from videospectra.server import create_app

    if host == "0.0.0.0":
        logger.warning(
            "binding to 0.0.0.0 exposes the dashboard on all interfaces. "
            "Use --host 127.0.0.1 for loopback only (default)."
        )

    app = create_app(session_factory)

    if open_browser:
        url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("webbrowser.open failed: %s", exc)

    uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_serve(args: argparse.Namespace) -> None:
    factory = _load_setup_module(Path(args.setup))
    _launch_server(factory, host=args.host, port=args.port, open_browser=args.open)


def _cmd_demo(args: argparse.Namespace) -> None:
    print(
        "videospectra demo: color-histogram embedder. For install verification, "
        "not real video understanding.",
        file=sys.stderr,
    )
    factory = _build_demo_session_factory()
    _launch_server(factory, host=args.host, port=args.port, open_browser=args.open)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videospectra", description="videospectra CLI")
    parser.add_argument("--version", action="version", version=f"videospectra {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the dashboard with a custom setup file")
    serve.add_argument(
        "--setup",
        required=True,
        help="path to a Python file defining `make_session() -> Session`",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--open",
        action="store_true",
        help="open the dashboard in the default browser on launch",
    )
    serve.set_defaults(func=_cmd_serve)

    demo = sub.add_parser("demo", help="run the dashboard with the bundled ColorHistogramEmbedder")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument(
        "--open",
        action="store_true",
        help="open the dashboard in the default browser on launch",
    )
    demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
