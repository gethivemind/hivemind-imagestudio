from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagestudio.config import Settings
from imagestudio.engine.fake import FakeEngine
from imagestudio.hardware import Gpu, Hardware
from imagestudio.registry import FLUX_SCHNELL, QWEN_IMAGE, ModelSpec, Variant
from imagestudio.service import Studio

GIB = 1024**3


def laptop(ram_gib: float = 16, cores: int = 10) -> Hardware:
    """The admin's machine: i5-1235U, 16 GB, Intel integrated graphics, no CUDA."""
    return Hardware(
        os="Windows",
        arch="AMD64",
        cpu_name="12th Gen Intel(R) Core(TM) i5-1235U",
        physical_cores=cores,
        logical_cores=cores + 2,
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB * 0.6),
        gpus=[Gpu("Intel(R) UHD Graphics", 2 * GIB, "integrated", shared_memory=True)],
    )


def workstation() -> Hardware:
    return Hardware(
        os="Linux",
        arch="x86_64",
        cpu_name="big",
        physical_cores=16,
        logical_cores=32,
        ram_total_bytes=64 * GIB,
        ram_available_bytes=48 * GIB,
        gpus=[Gpu("NVIDIA RTX 4090", 24 * GIB, "cuda")],
    )


def fake_install(settings: Settings, model: ModelSpec, variant: Variant) -> None:
    """Pretend a variant is downloaded: stub files plus a manifest entry. The size check is
    patched out by the autouse fixture below (real files are tens of GB)."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.manifest_file
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for f in variant.files:
        (settings.models_dir / f.filename).write_bytes(b"stub")
        manifest[f.filename] = f.sha256
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture(autouse=True)
def _no_size_check(monkeypatch):
    monkeypatch.setattr("imagestudio.models._size_matches", lambda _p, _s: True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture
def studio(settings: Settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q5"))
    st = Studio(settings=settings, engine=FakeEngine(), hardware=laptop())
    yield st
    st.close()


__all__ = ["FLUX_SCHNELL", "QWEN_IMAGE", "fake_install", "laptop", "workstation"]
