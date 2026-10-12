from __future__ import annotations

import pytest

from conftest import fake_install, laptop, workstation
from imagestudio.calibration import Calibration
from imagestudio.config import Settings
from imagestudio.models import ModelStore
from imagestudio.planner import PlanError, Planner, best_fitting_variant, memory_budget
from imagestudio.registry import FLUX_SCHNELL, MODELS, QWEN_IMAGE

GIB = 1024**3


def make_planner(settings: Settings, hw):
    store = ModelStore(settings)
    return Planner(hw, store, Calibration(settings.calibration_file, hw)), store


def test_memory_budget_ignores_integrated_gpu_counts_discrete():
    assert memory_budget(laptop(16)) == 14 * GIB
    assert memory_budget(workstation()) == (64 - 2 + 24) * GIB


def test_best_is_full_resolution_and_full_steps_on_the_laptop(settings):
    planner, _ = make_planner(settings, laptop())
    plan = planner.plan("flux-schnell", "best")
    assert (plan.width, plan.height, plan.steps) == (1024, 1024, 4)
    # q8 does not fit in 16 GB, so a lower quantisation is used and SAID so.
    assert plan.variant.name in ("q5", "q4")
    assert plan.fit_note and "memory fit" in plan.fit_note
    assert "resolution and steps are unchanged" in plan.fit_note


def test_best_on_a_workstation_uses_top_variant_with_no_fit_note(settings):
    planner, _ = make_planner(settings, workstation())
    plan = planner.plan("flux-schnell", "best")
    assert plan.variant.name == "q8"
    assert plan.fit_note is None


def test_presets_change_only_what_they_say(settings):
    planner, _ = make_planner(settings, workstation())
    best = planner.plan("flux-schnell", "best")
    balanced = planner.plan("flux-schnell", "balanced")
    fast = planner.plan("flux-schnell", "fast")
    assert best.steps == balanced.steps == fast.steps == 4  # schnell's full step count
    assert best.width > balanced.width > fast.width
    assert best.estimate.seconds > balanced.estimate.seconds > fast.estimate.seconds
    for p in (balanced, fast):
        assert p.preset.tradeoff  # trade-off stated in words


def test_explicit_size_overrides_preset_and_snaps_to_16(settings):
    planner, _ = make_planner(settings, workstation())
    plan = planner.plan("flux-schnell", "best", width=1000, height=777)
    assert (plan.width, plan.height) == (992, 768)
    with pytest.raises(PlanError):
        planner.plan("flux-schnell", "best", width=100)
    with pytest.raises(PlanError):
        planner.plan("flux-schnell", "best", height=4096)


def test_unknown_preset_or_model_is_an_error(settings):
    planner, _ = make_planner(settings, workstation())
    with pytest.raises(PlanError):
        planner.plan("flux-schnell", "turbo")
    with pytest.raises(PlanError):
        planner.plan("sdxl", "best")


def test_qwen_image_does_not_fit_the_laptop_and_says_so(settings):
    planner, _ = make_planner(settings, laptop())
    assert best_fitting_variant(QWEN_IMAGE, laptop(), 1024, 1024) is None
    with pytest.raises(PlanError, match="does not fit"):
        planner.plan("qwen-image", "best")


def test_auto_model_prefers_strongest_that_is_installed_and_fits(settings):
    planner, _ = make_planner(settings, workstation())
    # nothing installed: strongest that fits the workstation is qwen-image
    assert planner.resolve_model(None, 1024, 1024) is QWEN_IMAGE
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q8"))
    planner, _ = make_planner(settings, workstation())
    assert planner.resolve_model("auto", 1024, 1024) is FLUX_SCHNELL
    # on the laptop qwen never fits, so flux it is
    planner, _ = make_planner(settings, laptop())
    assert planner.resolve_model(None, 1024, 1024) is FLUX_SCHNELL


def test_installed_lower_variant_is_preferred_over_uninstalled_higher(settings):
    fake_install(settings, FLUX_SCHNELL, FLUX_SCHNELL.variant("q4"))
    planner, _ = make_planner(settings, laptop())
    plan = planner.plan("flux-schnell", "best")
    assert plan.variant.name == "q4"
    assert plan.installed
    assert any("not downloaded" in w for w in plan.warnings)  # said, not silent


def test_explicit_variant_that_does_not_fit_is_refused_with_numbers(settings):
    planner, _ = make_planner(settings, laptop())
    with pytest.raises(PlanError, match="GiB") as e:
        planner.plan("flux-schnell", "best", variant_name="q8")
    assert "budget" in str(e.value)


def test_estimate_is_labelled_uncalibrated_then_learns(settings):
    hw = laptop()
    cal = Calibration(settings.calibration_file, hw)
    v = FLUX_SCHNELL.variant("q5")
    e0 = cal.estimate(FLUX_SCHNELL, "q5", v.download_bytes, 1024, 1024, 4, 1.0)
    assert not e0.calibrated and "uncalibrated" in e0.basis
    assert 120 < e0.seconds < 1200  # minutes, as the ADR says, not seconds or hours
    cal.record(FLUX_SCHNELL, "q5", v.download_bytes, 1024, 1024, 4, 1.0, 60.0, 400.0)
    cal2 = Calibration(settings.calibration_file, hw)  # persisted
    e1 = cal2.estimate(FLUX_SCHNELL, "q5", v.download_bytes, 1024, 1024, 4, 1.0)
    assert e1.calibrated
    assert e1.seconds == pytest.approx(460.0)
    e_small = cal2.estimate(FLUX_SCHNELL, "q5", v.download_bytes, 512, 512, 4, 1.0)
    assert e_small.seconds == pytest.approx(60 + 100.0)


def test_every_catalogue_file_is_pinned_and_permissive():
    for m in MODELS.values():
        assert m.licence == "Apache-2.0"
        for v in m.variants:
            for f in v.files:
                assert len(f.sha256) == 64 and int(f.sha256, 16) >= 0
                assert f.size > 0
                assert f.licence in ("Apache-2.0", "MIT", "BSD-3-Clause")
                assert "FLUX.1-dev" not in f.url


def test_advanced_overrides_apply_and_are_validated(settings):
    planner, _ = make_planner(settings, workstation())
    plan = planner.plan("flux-schnell", "best", steps=8, cfg_scale=1.5, sampler="euler_a")
    assert (plan.steps, plan.cfg_scale, plan.sampler) == (8, 1.5, "euler_a")
    assert any("step-distilled" in w for w in plan.warnings)
    base = planner.plan("flux-schnell", "best")
    assert plan.estimate.seconds > base.estimate.seconds  # more steps and CFG cost time
    for bad in ({"steps": 0}, {"steps": 101}, {"cfg_scale": -1}, {"sampler": "magic"}):
        with pytest.raises(PlanError):
            planner.plan("flux-schnell", "best", **bad)


def test_low_free_ram_turns_on_memory_mapping(settings):
    hw = laptop()
    hw.ram_available_bytes = 2 * GIB
    planner, _ = make_planner(settings, hw)
    plan = planner.plan("flux-schnell", "best")
    assert plan.low_memory and any("memory-mapped" in w for w in plan.warnings)
    hw.ram_available_bytes = 14 * GIB
    planner, _ = make_planner(settings, hw)
    assert not planner.plan("flux-schnell", "best").low_memory


def test_load_meter_labels(settings):
    planner, _ = make_planner(settings, workstation())
    plan = planner.plan("flux-schnell", "best")
    assert plan.load.label == "comfortable" and plan.load.score < 70
    hw = laptop()
    hw.ram_available_bytes = 2 * GIB
    planner, _ = make_planner(settings, hw)
    plan = planner.plan("flux-schnell", "best")
    assert plan.load.label == "will swap" and plan.load.score <= 100
    assert plan.to_dict()["load"]["meaning"]


def test_over_budget_is_locked_unless_allowed(settings):
    planner, _ = make_planner(settings, laptop())
    with pytest.raises(PlanError, match="locked"):
        planner.plan("flux-schnell", "best", variant_name="q8")
    with pytest.raises(PlanError, match="locked"):
        planner.plan("qwen-image", "best")
    hw = laptop()
    store = ModelStore(settings)
    unlocked = Planner(
        hw, store, Calibration(settings.calibration_file, hw), allow_over_budget=True
    )
    plan = unlocked.plan("flux-schnell", "best", variant_name="q8")
    assert plan.over_budget and plan.load.label == "over budget"
    plan = unlocked.plan("qwen-image", "best")
    assert plan.over_budget and plan.variant.name == "q4"


def test_hardware_cache_is_reused_but_available_ram_is_live(tmp_path, monkeypatch):
    from imagestudio import hardware as hwmod

    calls = []

    def fake_probe():
        calls.append(1)
        return laptop()

    monkeypatch.setattr(hwmod, "_probe", fake_probe)
    cache = tmp_path / "hardware.json"
    a = hwmod.detect(cache)
    b = hwmod.detect(cache)
    assert len(calls) == 1 and cache.exists()
    assert a.cpu_name == b.cpu_name and b.gpus[0].name == a.gpus[0].name
    assert b.ram_available_bytes > 0


def test_small_images_have_a_per_step_floor(settings):
    hw = laptop()
    cal = Calibration(settings.calibration_file, hw)
    v = FLUX_SCHNELL.variant("q5")
    cal.record(FLUX_SCHNELL, "q5", v.download_bytes, 1024, 1024, 4, 1.0, 60.0, 400.0)
    e256 = cal.estimate(FLUX_SCHNELL, "q5", v.download_bytes, 256, 256, 4, 1.0)
    e512 = cal.estimate(FLUX_SCHNELL, "q5", v.download_bytes, 512, 512, 4, 1.0)
    k = cal.coefficients(FLUX_SCHNELL, "q5", v.download_bytes).seconds_per_step_megapixel
    # 256² is 1/16 of the pixels but is estimated at the floor (0.2 MP)
    assert e256.seconds - 60 == pytest.approx(4 * 0.2 * k, rel=0.01)
    assert e512.seconds - 60 == pytest.approx(4 * 0.262144 * k, rel=0.01)
