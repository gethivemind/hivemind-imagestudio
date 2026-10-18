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
    """Write the expanded canvas and its mask. Returns (width, height) of the canvas."""
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
    # mask: white = paint, black = keep; feathered a little into the original
    mask = Image.new("L", (cw, ch), 255)
    inset = max(0, min(feather, w // 4, h // 4))
    keep = Image.new("L", (max(1, w - 2 * inset), max(1, h - 2 * inset)), 0)
    mask.paste(keep, (x0 + inset, y0 + inset))
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
    return cw, ch


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
