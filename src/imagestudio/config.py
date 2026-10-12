"""Paths and settings. Everything the studio writes lives under one data directory."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "hivemind-imagestudio"
ENV_DATA_DIR = "IMAGESTUDIO_DATA_DIR"
ENV_SD_BIN = "IMAGESTUDIO_SD_BIN"
ENV_SD_FLAVOR = "IMAGESTUDIO_SDCPP_FLAVOR"
ENV_THREADS = "IMAGESTUDIO_THREADS"
ENV_ALLOW_OVER_BUDGET = "IMAGESTUDIO_ALLOW_OVER_BUDGET"


def default_data_dir() -> Path:
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


@dataclass(frozen=True)
class Settings:
    """Resolved locations. Create with ``Settings.load()`` or pass a ``data_dir`` in tests."""

    data_dir: Path = field(default_factory=default_data_dir)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def tools_dir(self) -> Path:
        return self.data_dir / "tools"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def calibration_file(self) -> Path:
        return self.data_dir / "calibration.json"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.json"

    def config(self) -> dict:
        """User settings from config.json in the data dir (missing file = {})."""
        try:
            return json.loads(self.config_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @property
    def allow_over_budget(self) -> bool:
        """The unlock: run settings the memory budget says will not fit. Off by default;
        enabled only by the environment variable or an explicit config.json entry."""
        env = os.environ.get(ENV_ALLOW_OVER_BUDGET, "").lower()
        if env in ("1", "true", "yes"):
            return True
        return bool(self.config().get("allow_over_budget", False))

    @property
    def worker_lock_file(self) -> Path:
        return self.data_dir / "worker.lock"

    @property
    def profile_file(self) -> Path:
        return self.data_dir / "profile.json"

    @property
    def manifest_file(self) -> Path:
        return self.models_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.tools_dir, self.out_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls) -> Settings:
        return cls(data_dir=default_data_dir())
