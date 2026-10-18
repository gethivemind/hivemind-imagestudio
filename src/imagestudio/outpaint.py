"""Outpainting: extend a picture beyond its edges, or change its aspect ratio, with the
existing masked image-to-image path. The new area is pre-filled by mirroring and blurring the
nearest edge so the model starts from the right colours, and the mask is feathered a little
into the original so the seam blends. Pure Pillow; the engine does the painting."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_SIDE = 2048
RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 1.5,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
}


@dataclass(frozen=True)
class Padding:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @property
    def any(self) -> bool:
        return any(v > 0 for v in (self.left, self.top, self.right, self.bottom))


def snap16(v: float) -> int:
    return max(16, int(v) // 16 * 16)


def padding_for_ratio(width: int, height: int, ratio: str) -> Padding:
    """Grow the image (never crop) until it has the target ratio, centred."""
    if ratio not in RATIOS:
        raise ValueError(f"unknown ratio {ratio!r}; one of {', '.join(RATIOS)}")
    target = RATIOS[ratio]
    current = width / height
    if abs(current - target) < 0.01:
        return Padding()
    if current < target:  # too tall: add width
        new_w = round(height * target)
        extra = new_w - width
        return Padding(left=extra // 2, right=extra - extra // 2)
    new_h = round(width / target)
    extra = new_h - height
    return Padding(top=extra // 2, bottom=extra - extra // 2)


def padding_for_fraction(width: int, height: int, fraction: float, sides: str = "all") -> Padding:
    """Grow by a fraction of the image on the chosen sides (``all``, ``horizontal``,
    ``vertical``, ``left``, ``right``, ``top``, ``bottom``)."""
    fraction = max(0.05, min(1.0, fraction))
    dx, dy = int(width * fraction), int(height * fraction)
    s = sides.lower()
    return Padding(
        left=dx if s in ("all", "horizontal", "left") else 0,
        right=dx if s in ("all", "horizontal", "right") else 0,
        top=dy if s in ("all", "vertical", "top") else 0,
        bottom=dy if s in ("all", "vertical", "bottom") else 0,
    )


def fit_to_engine(width: int, height: int, pad: Padding) -> tuple[int, int, float]:
    """The expanded size snapped to multiples of 16 and at most MAX_SIDE; returns the scale
    applied to the whole canvas (1.0 when it fits)."""
    w, h = width + pad.left + pad.right, height + pad.top + pad.bottom
    scale = min(1.0, MAX_SIDE / max(w, h))
    return snap16(w * scale), snap16(h * scale), scale


def expand(source: Path, pad: Padding, out_image: Path, out_mask: Path, feather: int = 24):
    """Write the expanded canvas and its mask. Returns the canvas size and where the original
    sits on it: (width, height, (x, y, w, h))."""
    from PIL import Image, ImageFilter, ImageOps

    with Image.open(source) as src:
        im = src.convert("RGB")
    w, h = im.size
    cw, ch, scale = fit_to_engine(w, h, pad)
    if scale < 1.0:
        im = im.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.LANCZOS)
        w, h = im.size
        pad = Padding(*(int(v * scale) for v in (pad.left, pad.top, pad.right, pad.bottom)))
    # the original sits at the padding offset; whatever rounding left over goes right/bottom
    x0, y0 = min(pad.left, cw - w), min(pad.top, ch - h)
    x0, y0 = max(0, x0), max(0, y0)
    canvas = Image.new("RGB", (cw, ch))
    # pre-fill: mirror the picture outward in both axes, then blur hard, so the new area
    # continues the edge colours instead of starting from black
    big = Image.new("RGB", (cw, ch))
    tiles_x = range(x0 - w, cw, w) if w else []
    for ty, flip_y in _mirror_positions(y0, h, ch):
        for tx, flip_x in _mirror_positions(x0, w, cw):
            tile = im
            if flip_x:
                tile = ImageOps.mirror(tile)
            if flip_y:
                tile = ImageOps.flip(tile)
            big.paste(tile, (tx, ty))
    del tiles_x
    canvas.paste(big.filter(ImageFilter.GaussianBlur(max(8, min(w, h) // 20))), (0, 0))
    canvas.paste(im, (x0, y0))
    # mask: white = paint, black = keep; feathered into the original only on the sides that
    # grew, so untouched edges stay exactly as they were
    mask = Image.new("L", (cw, ch), 255)
    inset = max(0, min(max(feather, min(w, h) // 12), w // 4, h // 4))
    il = inset if pad.left else 0
    it = inset if pad.top else 0
    ir = inset if pad.right else 0
    ib = inset if pad.bottom else 0
    keep = Image.new("L", (max(1, w - il - ir), max(1, h - it - ib)), 0)
    mask.paste(keep, (x0 + il, y0 + it))
    if inset:
        mask = mask.filter(ImageFilter.GaussianBlur(inset / 2))
        # the far outside must stay fully white after the blur
        outer = Image.new("L", (cw, ch), 255)
        inner_box = Image.new("L", (w, h), 0)
        outer.paste(inner_box, (x0, y0))
        mask = Image.composite(outer, mask, outer)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_image, "PNG")
    mask.save(out_mask, "PNG")
    return cw, ch, (x0, y0, w, h)


def _mirror_positions(start: int, size: int, limit: int):
    """Tile positions covering [0, limit) around the original at ``start``: each step away
    flips the tile, so edges meet their own reflection."""
    out = []
    n = 0
    pos = start
    while pos + size > 0:
        out.append((pos, n % 2 == 1))
        pos -= size
        n += 1
    n = 1
    pos = start + size
    while pos < limit:
        out.append((pos, n % 2 == 1))
        pos += size
        n += 1
    return out


def finish(
    result: Path, canvas: Path, region: tuple[int, int, int, int], feather: int = 48
) -> None:
    """After the engine: match the painted canvas's colours to the original region (the model
    drifts cooler and flatter on the pre-filled area), then put the original pixels back
    exactly under a wide feathered mask. Rewrites ``result`` in place."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    x, y, w, h = region
    with Image.open(result) as r, Image.open(canvas) as c:
        if r.size != c.size or w < 8 or h < 8:
            return  # not an outpaint result we can finish (e.g. a test engine's stub image)
        out = np.asarray(r.convert("RGB"), dtype=np.float32)
        orig = np.asarray(c.convert("RGB"), dtype=np.float32)[y : y + h, x : x + w]
    painted_region = out[y : y + h, x : x + w]
    # per-channel mean/std transfer: what the model did to the kept area is what it did to
    # the whole picture, so undo that drift everywhere
    src_m, src_s = (
        painted_region.reshape(-1, 3).mean(0),
        painted_region.reshape(-1, 3).std(0) + 1e-3,
    )
    dst_m, dst_s = orig.reshape(-1, 3).mean(0), orig.reshape(-1, 3).std(0) + 1e-3
    gain = np.clip(dst_s / src_s, 0.6, 1.6)
    out = np.clip((out - src_m) * gain + dst_m, 0, 255)
    matched = Image.fromarray(out.astype(np.uint8))
    with Image.open(canvas) as c:
        original = c.convert("RGB").crop((x, y, x + w, y + h))
    inset = max(2, min(max(feather, min(w, h) // 12), w // 4, h // 4))
    mask = Image.new("L", matched.size, 0)
    ImageDraw.Draw(mask).rectangle((x + inset, y + inset, x + w - inset, y + h - inset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(inset / 2))
    full = Image.new("RGB", matched.size)
    full.paste(original, (x, y))
    final = Image.composite(full, matched, mask)
    final.save(result, "PNG")
