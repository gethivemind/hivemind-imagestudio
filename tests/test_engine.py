"""The stable-diffusion.cpp wrapper, tested against a stand-in script that prints the
engine's progress format. No models, no network, no real binary."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from imagestudio.config import ENV_SD_BIN, Settings
from imagestudio.engine.base import EngineError, GenerateSpec
from imagestudio.engine.fake import tiny_png
from imagestudio.engine.sdcpp import SdCppEngine
from imagestudio.registry import SDCPP_ASSETS, find_engine_asset

FAKE_SD = textwrap.dedent(
    """
    import sys, time
    args = sys.argv[1:]
    out = args[args.index("-o") + 1]
    steps = int(args[args.index("--steps") + 1])
    if "--fail" in args:
        print("[ERROR] something broke", flush=True)
        sys.exit(3)
    print("[INFO ] loading model from 'x.gguf'", flush=True)
    print("[INFO ] loading tensors completed, taking 1.50s", flush=True)
    for i in range(1, steps + 1):
        sys.stdout.write(f"  |{'=' * i}>{' ' * (steps - i)}| {i}/{steps} - 0.10s/it\\r")
        sys.stdout.flush()
        time.sleep(0.02)
    print("", flush=True)
    print("[INFO ] sampling completed, taking 0.40s", flush=True)
    print("[INFO ] decode_first_stage completed, taking 0.05s", flush=True)
    with open(out, "wb") as f:
        f.write(bytes.fromhex(sys.argv[-1]))
    print("[INFO ] save result PNG image to '" + out + "'", flush=True)
    """
)


@pytest.fixture
def fake_sd(tmp_path: Path, monkeypatch):
    script = tmp_path / "fake_sd.py"
    script.write_text(FAKE_SD, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "sd-cli.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "sd-cli"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    monkeypatch.setenv(ENV_SD_BIN, str(launcher))
    return launcher


def spec(tmp_path: Path, **over) -> GenerateSpec:
    base = {
        "prompt": "a cat",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "steps": 4,
        "cfg_scale": 1.0,
        "seed": 7,
        "sampler": "euler",
        "output": tmp_path / "out" / "img.png",
        "component_paths": {"--diffusion-model": Path("m.gguf"), "--vae": Path("ae.safetensors")},
        "extra_args": (tiny_png().hex(),),  # the fake script writes argv[-1] as the image
    }
    base.update(over)
    return GenerateSpec(**base)


def test_build_args_follow_sdcpp_cli(tmp_path, fake_sd):
    engine = SdCppEngine(Settings(data_dir=tmp_path / "data"))
    args = engine.build_args(spec(tmp_path, negative_prompt="blurry", threads=4))
    joined = " ".join(args)
    assert args[0] == str(fake_sd)
    for flag, value in [
        ("-p", "a cat"),
        ("-n", "blurry"),
        ("-W", "512"),
        ("-H", "512"),
        ("--steps", "4"),
        ("--cfg-scale", "1.0"),
        ("--sampling-method", "euler"),
        ("-s", "7"),
        ("-t", "4"),
        ("--diffusion-model", "m.gguf"),
        ("--vae", "ae.safetensors"),
    ]:
        assert args[args.index(flag) + 1] == value, (flag, joined)


def test_generate_parses_progress_and_timings(tmp_path, fake_sd):
    engine = SdCppEngine(Settings(data_dir=tmp_path / "data"))
    seen = []
    result = engine.generate(spec(tmp_path), seen.append)
    assert result.output.exists() and result.output.read_bytes().startswith(b"\x89PNG")
    phases = [p.phase for p in seen]
    assert phases[0] == "loading"
    assert "sampling" in phases and "decoding" in phases and "saving" in phases
    steps = [(p.step, p.total_steps) for p in seen if p.phase == "sampling"]
    assert steps[-1] == (4, 4)
    # load time is wall time until the first sampling step, not the engine's log value
    assert 0 < result.load_seconds < 10  # wall time; generous, the suite often runs under load
    assert result.sampling_seconds == pytest.approx(0.4)
    assert result.total_seconds > 0


def test_engine_failure_surfaces_log_tail(tmp_path, fake_sd):
    engine = SdCppEngine(Settings(data_dir=tmp_path / "data"))
    with pytest.raises(EngineError, match="code 3") as e:
        engine.generate(spec(tmp_path, extra_args=("--fail", "00")))
    assert "something broke" in str(e.value)


def test_binary_lookup_order(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_SD_BIN, raising=False)
    monkeypatch.setenv("PATH", "")
    engine = SdCppEngine(Settings(data_dir=tmp_path / "data"), flavor="cpu")
    assert engine.binary() is None and not engine.available()
    a = engine.asset()
    if a is not None:
        exe = engine.install_dir / a.exe
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
        assert engine.binary() == exe


def test_pinned_assets_cover_the_three_desktop_platforms():
    assert find_engine_asset("win32", "AMD64", "cpu") is not None
    assert find_engine_asset("win32", "AMD64", "vulkan") is not None
    assert find_engine_asset("linux", "x86_64", "cpu") is not None
    assert find_engine_asset("darwin", "arm64", "cpu") is not None
    assert find_engine_asset("linux", "riscv64", "cpu") is None
    for a in SDCPP_ASSETS:
        assert len(a.sha256) == 64 and a.size > 0 and a.url.startswith("https://github.com/")


def test_preview_flags_are_passed_when_requested(tmp_path, fake_sd):
    engine = SdCppEngine(Settings(data_dir=tmp_path / "data"))
    args = engine.build_args(spec(tmp_path, preview_path=tmp_path / "p.png"))
    assert args[args.index("--preview") + 1] == "proj"
    assert args[args.index("--preview-path") + 1] == str(tmp_path / "p.png")
    assert "--preview" not in engine.build_args(spec(tmp_path))


def test_engine_backend_setting_and_switching(settings, monkeypatch):
    import json

    from imagestudio.config import ENV_SD_FLAVOR
    from imagestudio.engine.sdcpp import SdCppEngine
    from imagestudio.service import Studio, StudioError

    monkeypatch.delenv(ENV_SD_FLAVOR, raising=False)
    assert settings.engine_flavor == "cpu"
    settings.config_file.parent.mkdir(parents=True, exist_ok=True)
    settings.config_file.write_text(json.dumps({"engine_backend": "vulkan"}), encoding="utf-8")
    assert settings.engine_flavor == "vulkan"
    assert SdCppEngine(settings).flavor == "vulkan"
    assert SdCppEngine(settings).install_dir.name.endswith("-vulkan")
    monkeypatch.setenv(ENV_SD_FLAVOR, "cpu")
    assert settings.engine_flavor == "cpu"  # the environment wins
    monkeypatch.delenv(ENV_SD_FLAVOR, raising=False)
    settings.config_file.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("IMAGESTUDIO_SD_BIN", "")
    studio = Studio(settings=settings, engine=SdCppEngine(settings), start_worker=False)
    try:
        cfg = studio.get_config()
        assert cfg["engine_backend"] == "cpu"
        flavors = {b["flavor"]: b for b in cfg["engine_backends"]}
        assert set(flavors) == {"cpu", "vulkan"} and flavors["cpu"]["active"]
        rows = [c for c in studio.components() if c["kind"] == "engine"]
        names = {r["name"] for r in rows}
        assert "vision" in names and any(n.startswith("image:") for n in names)
        # switching to a build that is not installed is refused with a clear message
        with pytest.raises(StudioError, match="not installed"):
            studio.set_engine_backend("vulkan")
        with pytest.raises(StudioError, match="unknown engine backend"):
            studio.set_engine_backend("cuda")
        assert studio._engine_for("image") is studio.engine
        assert studio._engine_for("image:vulkan").flavor == "vulkan"
    finally:
        studio.close()


def test_build_args_reference_images_replace_init(tmp_path, fake_sd):
    settings = Settings(data_dir=tmp_path / "data")
    eng = SdCppEngine(settings)
    spec = GenerateSpec(
        prompt="make it night",
        negative_prompt="",
        width=512,
        height=512,
        steps=12,
        cfg_scale=2.5,
        seed=1,
        sampler="euler",
        output=tmp_path / "o.png",
        component_paths={},
        init_image=tmp_path / "src.png",
        strength=0.45,
        ref_images=(tmp_path / "src.png",),
    )
    args = eng.build_args(spec)
    assert args[args.index("-r") + 1] == str(tmp_path / "src.png")
    assert "-i" not in args and "--strength" not in args
    spec.ref_images = ()
    args = eng.build_args(spec)
    assert "-r" not in args and args[args.index("-i") + 1] == str(tmp_path / "src.png")
