"""Parameters inside the prompt, Midjourney style: ``a fox at dawn --seed 7 --size 1024x768``.

The studio parses them out, uses them as settings, and sends the model only the words. A
finished job can be written back as one reproducible line (``reproducible_prompt``), so
copying it from the gallery and pasting it into the prompt box recreates the same image.

Grammar: ``--name value`` tokens anywhere in the text; ``--name`` alone for booleans. Names
are case-insensitive. Unknown names are left in the prompt untouched and reported so the UI
can point at them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"(?:(?<=\s)|^)--([a-zA-Z][\w-]*)(?:\s+(?!--)(\S+))?")

ALIASES = {
    "seed": "seed",
    "s": "seed",
    "size": "size",
    "ar": "aspect",
    "aspect": "aspect",
    "w": "width",
    "width": "width",
    "h": "height",
    "height": "height",
    "preset": "preset",
    "p": "preset",
    "steps": "steps",
    "cfg": "cfg_scale",
    "guidance": "cfg_scale",
    "sampler": "sampler",
    "model": "model",
    "variant": "variant",
    "q": "variant",
    "no": "negative_prompt",
    "neg": "negative_prompt",
    "negative": "negative_prompt",
    "threads": "threads",
    "tiling": "vae_tiling",
    "vae-tiling": "vae_tiling",
    "from": "init",
    "n": "count",
    "count": "count",
    "batch": "count",
    "init": "init",
    "strength": "strength",
    "str": "strength",
}

# how each parameter is written back
_ORDER = (
    "preset",
    "size",
    "seed",
    "model",
    "variant",
    "steps",
    "cfg_scale",
    "sampler",
    "negative_prompt",
)


@dataclass
class Parsed:
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "params": self.params,
            "unknown": self.unknown,
            "errors": self.errors,
        }


def _int(name: str, raw: str, lo: int, hi: int, errors: list[str]) -> int | None:
    try:
        v = int(raw)
    except ValueError:
        errors.append(f"--{name} needs a whole number, got {raw!r}")
        return None
    if not lo <= v <= hi:
        errors.append(f"--{name} must be between {lo} and {hi}, got {v}")
        return None
    return v


def parse(text: str) -> Parsed:
    """Split ``text`` into the words for the model and the parameters."""
    params: dict[str, Any] = {}
    unknown: list[str] = []
    errors: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _TOKEN.finditer(text):
        name, raw = m.group(1).lower(), m.group(2)
        key = ALIASES.get(name)
        if key is None:
            unknown.append(name)
            continue
        spans.append(m.span())
        if key == "vae_tiling":
            params[key] = raw is None or raw.lower() in ("1", "on", "true", "yes")
            if raw is not None and raw.lower() not in (
                "1",
                "on",
                "true",
                "yes",
                "0",
                "off",
                "false",
                "no",
            ):
                # the value was not ours: give it back to the prompt
                spans[-1] = (m.start(), m.start(1) + len(name))
            continue
        if raw is None:
            errors.append(f"--{name} needs a value")
            continue
        if key == "seed":
            v = _int(name, raw, 0, 2**31 - 1, errors)
            if v is not None:
                params["seed"] = v
        elif key == "size":
            mm = re.fullmatch(r"(\d+)\s*[x×*]\s*(\d+)", raw, re.IGNORECASE)
            if not mm:
                errors.append(f"--size wants WIDTHxHEIGHT, got {raw!r}")
            else:
                w = _int("size", mm.group(1), 256, 2048, errors)
                h = _int("size", mm.group(2), 256, 2048, errors)
                if w is not None and h is not None:
                    params["width"], params["height"] = w, h
        elif key == "aspect":
            mm = re.fullmatch(r"(\d+(?:\.\d+)?)[:/](\d+(?:\.\d+)?)", raw)
            if not mm:
                errors.append(f"--ar wants W:H like 16:9, got {raw!r}")
            else:
                params["aspect"] = (float(mm.group(1)), float(mm.group(2)))
        elif key in ("width", "height"):
            v = _int(name, raw, 256, 2048, errors)
            if v is not None:
                params[key] = v
        elif key == "steps":
            v = _int(name, raw, 1, 100, errors)
            if v is not None:
                params["steps"] = v
        elif key == "threads":
            v = _int(name, raw, 1, 256, errors)
            if v is not None:
                params["threads"] = v
        elif key == "count":
            v = _int(name, raw, 1, 16, errors)
            if v is not None:
                params["count"] = v
        elif key == "cfg_scale":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"--{name} needs a number, got {raw!r}")
            else:
                if 0 <= v <= 20:
                    params["cfg_scale"] = v
                else:
                    errors.append(f"--{name} must be between 0 and 20, got {v}")
        elif key == "strength":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"--{name} needs a number between 0 and 1, got {raw!r}")
            else:
                if 0.0 <= v <= 1.0:
                    params["strength"] = v
                else:
                    errors.append(f"--{name} must be between 0 and 1, got {v}")
        elif key == "init":
            params["init"] = raw  # a job id or a path; resolved by the studio
        elif key == "strength":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"--{name} needs a number between 0 and 1, got {raw!r}")
            else:
                if 0.0 <= v <= 1.0:
                    params["strength"] = v
                else:
                    errors.append(f"--{name} must be between 0 and 1, got {v}")
        elif key == "init":
            params["init"] = raw  # a job id or a path; resolved by the studio
        elif key == "strength":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"--{name} needs a number between 0 and 1, got {raw!r}")
            else:
                if 0.0 <= v <= 1.0:
                    params["strength"] = v
                else:
                    errors.append(f"--{name} must be between 0 and 1, got {v}")
        elif key == "init":
            params["init"] = raw  # a job id or a path; resolved by the studio
        elif key == "preset":
            if raw.lower() in ("best", "balanced", "fast"):
                params["preset"] = raw.lower()
            else:
                errors.append(f"--preset must be best, balanced or fast, got {raw!r}")
        elif key == "variant":
            if raw.lower() in ("q8", "q6", "q5", "q4"):
                params["variant"] = raw.lower()
            else:
                errors.append(f"--{name} must be q8, q6, q5 or q4, got {raw!r}")
        elif key == "negative_prompt":
            # --no takes everything up to the next --param
            end = text.find(" --", m.end(1))
            value = text[m.start(2) : end if end >= 0 else len(text)].strip()
            spans[-1] = (m.start(), end if end >= 0 else len(text))
            params["negative_prompt"] = value
        else:  # sampler, model: free strings validated by the planner
            params[key] = raw
    # remove the recognised tokens from the text, right to left
    clean = text
    for a, b in sorted(spans, reverse=True):
        clean = clean[:a] + clean[b:]
    clean = re.sub(r"\s{2,}", " ", clean).strip().strip(",").strip()
    # resolve aspect against width/height once we know a base size (handled by caller)
    return Parsed(prompt=clean, params=params, unknown=unknown, errors=errors)


def apply_aspect(params: dict[str, Any], base_width: int, base_height: int) -> None:
    """Turn ``aspect`` into width/height keeping the pixel count of the base size."""
    if "aspect" not in params:
        return
    a, b = params.pop("aspect")
    px = base_width * base_height
    w = int(round((px * a / b) ** 0.5 / 16) * 16)
    h = int(round((px * b / a) ** 0.5 / 16) * 16)
    params.setdefault("width", max(256, min(2048, w)))
    params.setdefault("height", max(256, min(2048, h)))


def reproducible(job: Any) -> str:
    """One line that recreates the job when pasted into the prompt box."""
    parts = [job.prompt]
    parts.append(f"--preset {job.preset}")
    parts.append(f"--size {job.width}x{job.height}")
    parts.append(f"--seed {job.seed}")
    parts.append(f"--model {job.model}")
    parts.append(f"--variant {job.variant}")
    parts.append(f"--steps {job.steps}")
    parts.append(f"--cfg {job.cfg_scale:g}")
    if job.sampler != "euler":
        parts.append(f"--sampler {job.sampler}")
    src = getattr(job, "source_job", None) or getattr(job, "input_path", None)
    if src and getattr(job, "tool", "generate") in ("generate", "variations", "edit"):
        parts.append(f"--from {src} --strength {getattr(job, 'strength', 0.6):g}")
    if job.negative_prompt:
        parts.append(f"--no {job.negative_prompt}")
    return " ".join(parts)


_BATCH_TOKENS = re.compile(r"(?:(?<=\s)|^)--(?:n|count|batch|seed|s)\s+\S+", re.IGNORECASE)


def without_batch_and_seed(text: str) -> str:
    """The prompt text with any --n/--count and --seed parameters removed, for the members of a
    batch, whose seeds are assigned one by one."""
    return re.sub(r"\s{2,}", " ", _BATCH_TOKENS.sub("", text)).strip()
