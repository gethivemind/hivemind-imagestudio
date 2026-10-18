"""Outpainting: extend a picture beyond its edges, or change its aspect ratio, with the
existing masked image-to-image path. The new area is pre-filled by mirroring and blurring the
nearest edge so the model starts from the right colours, and the mask is feathered a little
into the original so the seam blends. Pure Pillow; the engine does the painting."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
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


def expand(source: Path, pad: Padding, out_image: Path, out_mask: Path, overlap: float = 0.25):
    """Write the expanded canvas and its mask. Returns the canvas size and where the original
    sits on it: (width, height, (x, y, w, h)).

    The new area is pre-filled by stretching the nearest edge outward (so a sky gradient keeps
    its colours instead of starting from grey), and the mask repaints a wide band of the
    original along each grown side (``overlap`` of the original's size) so the model draws the
    transition itself; a hard edge is what makes outpaints look pasted."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    with Image.open(source) as src:
        im = src.convert("RGB")
    w, h = im.size
    cw, ch, scale = fit_to_engine(w, h, pad)
    if scale < 1.0:
        im = im.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.LANCZOS)
        w, h = im.size
        pad = Padding(*(int(v * scale) for v in (pad.left, pad.top, pad.right, pad.bottom)))
    x0 = max(0, min(pad.left, cw - w))
    y0 = max(0, min(pad.top, ch - h))
    a = np.asarray(im, dtype=np.uint8)
    # stretch outward: columns first, then rows, so corners take the nearest edge
    left = np.repeat(a[:, :1, :], x0, axis=1) if x0 else np.zeros((h, 0, 3), np.uint8)
    right_n = max(0, cw - x0 - w)
    right = np.repeat(a[:, -1:, :], right_n, axis=1) if right_n else np.zeros((h, 0, 3), np.uint8)
    row = np.concatenate([left, a, right], axis=1)
    top = np.repeat(row[:1, :, :], y0, axis=0) if y0 else np.zeros((0, cw, 3), np.uint8)
    bottom_n = max(0, ch - y0 - h)
    bottom = (
        np.repeat(row[-1:, :, :], bottom_n, axis=0) if bottom_n else np.zeros((0, cw, 3), np.uint8)
    )
    canvas = Image.fromarray(np.concatenate([top, row, bottom], axis=0))
    # soften the stretched area a little (streaks become gradients); the original stays sharp.
    # Mirrored tiles were tried as a pre-fill and made the model paint a second subject
    # (a face above the face); plain stretching fails softly instead, as blur.
    canvas = canvas.filter(ImageFilter.GaussianBlur(max(4, min(w, h) // 40)))
    canvas.paste(im, (x0, y0))
    # mask: white = paint. Everything new, plus a band of the original along each grown side.
    band_x, band_y = int(w * overlap), int(h * overlap)
    kl = band_x if pad.left else 0
    kr = band_x if pad.right else 0
    kt = band_y if pad.top else 0
    kb = band_y if pad.bottom else 0
    mask = Image.new("L", (cw, ch), 255)
    keep_w, keep_h = max(1, w - kl - kr), max(1, h - kt - kb)
    ImageDraw.Draw(mask).rectangle(
        (x0 + kl, y0 + kt, x0 + kl + keep_w - 1, y0 + kt + keep_h - 1), fill=0
    )
    blur = max(2, min(band_x if (kl or kr) else band_y, band_y if (kt or kb) else band_x) // 4)
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    outer = Image.new("L", (cw, ch), 255)
    outer.paste(Image.new("L", (w, h), 0), (x0, y0))
    mask = Image.composite(outer, mask, outer)  # the new area is always fully white
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
    result: Path, canvas: Path, region: tuple[int, int, int, int], overlap: float = 0.25
) -> None:
    """After the engine: nudge the whole picture's colours towards the original (measured on
    the kept interior), then restore the original's interior exactly under a soft mask that
    reaches the band the model repainted. Rewrites ``result`` in place."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    x, y, w, h = region
    with Image.open(result) as r, Image.open(canvas) as c:
        if r.size != c.size or w < 8 or h < 8:
            return  # not an outpaint result we can finish (e.g. a test engine's stub image)
        out = np.asarray(r.convert("RGB"), dtype=np.float32)
        orig_full = c.convert("RGB")
        orig = np.asarray(orig_full, dtype=np.float32)[y : y + h, x : x + w]
    bx, by = int(w * overlap), int(h * overlap)
    inner = (slice(y + by, y + h - by), slice(x + bx, x + w - bx))
    painted_inner = out[inner]
    orig_inner = orig[by : h - by, bx : w - bx]
    if painted_inner.size and orig_inner.size:
        src_m = painted_inner.reshape(-1, 3).mean(0)
        src_s = painted_inner.reshape(-1, 3).std(0) + 1e-3
        dst_m = orig_inner.reshape(-1, 3).mean(0)
        dst_s = orig_inner.reshape(-1, 3).std(0) + 1e-3
        gain = np.clip(dst_s / src_s, 0.7, 1.4)
        shift = np.clip(dst_m - src_m * gain, -40, 40)
        out = np.clip(out * gain + shift, 0, 255)
    matched = Image.fromarray(out.astype(np.uint8))
    mask = Image.new("L", matched.size, 0)
    ImageDraw.Draw(mask).rectangle((x + bx, y + by, x + w - bx, y + h - by), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2, min(bx, by) // 2 or 2)))
    full = Image.new("RGB", matched.size)
    full.paste(orig_full.crop((x, y, x + w, y + h)), (x, y))
    Image.composite(full, matched, mask).save(result, "PNG")
