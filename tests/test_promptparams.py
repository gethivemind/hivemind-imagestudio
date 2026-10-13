from __future__ import annotations

from imagestudio.promptparams import (
    apply_aspect,
    parse,
    reproducible,
    without_batch_and_seed,
)


def test_parse_midjourney_style_params():
    p = parse(
        "a fox at dawn, oil painting --seed 7 --size 1024x768 --preset fast --cfg 1.5 --steps 8"
    )
    assert p.prompt == "a fox at dawn, oil painting"
    assert p.params == {
        "seed": 7,
        "width": 1024,
        "height": 768,
        "preset": "fast",
        "cfg_scale": 1.5,
        "steps": 8,
    }
    assert not p.errors and not p.unknown


def test_negative_prompt_takes_the_rest_until_next_param():
    p = parse("a cat --no blurry, text, watermark --seed 3")
    assert p.prompt == "a cat" and p.params["negative_prompt"] == "blurry, text, watermark"
    assert p.params["seed"] == 3


def test_slashes_and_dashes_in_text_are_left_alone():
    p = parse("1/2 cup of coffee, black/white photo, a well-known--rare bird")
    assert p.prompt.startswith("1/2 cup of coffee, black/white photo")
    assert p.params == {}


def test_unknown_and_invalid_params_are_reported():
    p = parse("x --foo 3 --seed abc --preset ultra")
    assert "foo" in p.unknown
    assert any("--seed" in e for e in p.errors) and any("--preset" in e for e in p.errors)
    assert "--foo 3" in p.prompt  # unknown tokens stay in the text


def test_aspect_keeps_pixel_count():
    p = parse("x --ar 16:9")
    apply_aspect(p.params, 1024, 1024)
    assert p.params["width"] % 16 == 0 and p.params["height"] % 16 == 0
    assert abs(p.params["width"] * p.params["height"] - 1024 * 1024) < 1024 * 1024 * 0.08
    assert p.params["width"] > p.params["height"]


def test_reproducible_round_trip():
    class J:
        prompt = "a lighthouse"
        preset = "best"
        width = 1024
        height = 768
        seed = 42
        model = "flux-schnell"
        variant = "q5"
        steps = 4
        cfg_scale = 1.0
        sampler = "euler"
        negative_prompt = "blurry"

    line = reproducible(J)
    p = parse(line)
    assert p.prompt == "a lighthouse"
    assert p.params["seed"] == 42 and p.params["width"] == 1024 and p.params["height"] == 768
    assert p.params["negative_prompt"] == "blurry" and p.params["variant"] == "q5"
    assert "--sampler" not in line  # default sampler is implied


def test_from_and_strength_params():
    p = parse("a copper teapot --from job_abc123 --strength 0.3")
    assert p.prompt == "a copper teapot" and p.params == {"init": "job_abc123", "strength": 0.3}
    assert any("--strength" in e for e in parse("x --strength 3").errors)


def test_batch_count_and_stripping():
    p = parse("a fox --n 4 --seed 9 --preset fast")
    assert p.params["count"] == 4 and p.params["seed"] == 9 and p.prompt == "a fox"
    assert parse("x --n 0").errors and parse("x --batch 17").errors
    assert without_batch_and_seed("a fox --n 4 --seed 9 --preset fast") == "a fox --preset fast"
    assert without_batch_and_seed("--count 2 a fox --s 3") == "a fox"
