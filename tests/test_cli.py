"""Tests for videospectra.cli — entrypoint and subcommands."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from videospectra import __version__
from videospectra import cli as cli_mod
from videospectra.cli import _build_demo_session_factory, _load_setup_module, build_parser


def _python() -> str:
    return sys.executable


class TestParser:
    def test_version_flag(self) -> None:
        result = subprocess.run(
            [_python(), "-m", "videospectra.cli", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_no_subcommand_exits_2(self) -> None:
        result = subprocess.run(
            [_python(), "-m", "videospectra.cli"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 2

    def test_serve_requires_setup(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve"])

    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--setup", "x.py"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765
        assert args.open is False

    def test_demo_defaults_loopback(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765


class TestSetupLoader:
    def test_missing_setup_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _load_setup_module(tmp_path / "does-not-exist.py")

    def test_setup_without_make_session_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_text("# no make_session here\n")
        with pytest.raises(SystemExit):
            _load_setup_module(bad)

    def test_setup_with_make_session_loads(self, tmp_path: Path) -> None:
        good = tmp_path / "good.py"
        good.write_text(
            "from videospectra.embedders import DummyEmbedder\n"
            "from videospectra.session import Session\n"
            "def make_session():\n"
            "    return Session(frame_embedder=DummyEmbedder.make_image())\n"
        )
        fn = _load_setup_module(good)
        assert callable(fn)
        # Build the session and check it constructed
        s = fn()
        # Don't call aclose without start; just check type
        assert s is not None

    def test_setup_raising_on_import_exits(self, tmp_path: Path) -> None:
        bad = tmp_path / "raises.py"
        bad.write_text("raise RuntimeError('boom')\n")
        with pytest.raises(SystemExit):
            _load_setup_module(bad)


class TestDemoFactory:
    def test_demo_factory_builds_session(self) -> None:
        factory = _build_demo_session_factory()
        session = factory()
        caps = session.capabilities()
        assert caps.spectral_analytics is True
        assert caps.clip_events is False
        assert caps.prompts is False


class TestNoTorchInDemo:
    def test_cli_module_does_not_import_torch_at_load(self) -> None:
        # The videospectra.cli module itself must not pull in torch at import.
        # We can't observe sys.modules cleanly inside rzen_venv (torch is
        # there for other reasons), so check the source file.
        src = Path(cli_mod.__file__).read_text()
        assert "import torch" not in src
        assert "from torch" not in src
