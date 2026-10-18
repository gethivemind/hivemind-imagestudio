"""The Presidium Image Studio mark, drawn with Pillow from the family's SVG geometry (ADR-065):
six seats around one table with the chair at the head; Image Studio puts a picture on the
table. Used for the app icon, home-screen PNGs, Store artwork and the in-app raster icons, so
every raster is generated from one definition and matches the SVGs in the brand folder."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from pathlib import Path

FAMILY = "Presidium"
PRODUCT = "Image Studio"
FULL_NAME = f"{FAMILY} {PRODUCT}"
VIOLET = "#5B4FD1"  # Presidium
TEAL = "#149A78"  # Image Studio
LIGHT = "#F4F4F5"
INK = "#15161A"

SEATS = ((53.1, 18.2), (58.3, 41.0), (43.7, 59.3), (20.3, 59.3), (5.7, 41.0), (10.9, 18.2))
CHAIR = (32.0, 8.0, 5.5)
SEAT_R = 3.6
FRAME = (21.0, 26.0, 22.0, 18.0, 3.0)  # x, y, w, h, rx
FRAME_STROKE = 3.0
MOUNTAIN = ((23, 42), (30, 34), (34, 38), (37, 35), (41, 40))
MOUNTAIN_STROKE = 2.6
SUN = (37.0, 31.0, 1.8)

MARK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="{label}">'
    "{tile}"
    + "".join(f'<circle cx="{x}" cy="{y}" r="3.6" fill="{{c}}"/>' for x, y in SEATS)
    + '<circle cx="32" cy="8" r="5.5" fill="{c}"/>'
    '<rect x="21" y="26" width="22" height="18" rx="3" fill="none" stroke="{c}" stroke-width="3"/>'
    '<path d="M23 42 L30 34 L34 38 L37 35 L41 40" fill="none" stroke="{c}" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="37" cy="31" r="1.8" fill="{c}"/></svg>'
)


def mark_svg(colour: str = TEAL, tile: str | None = None, label: str = FULL_NAME) -> str:
    """The mark as SVG text: ``tile`` colours a rounded square behind a light mark."""
    if tile:
        return MARK_SVG.format(
            label=label,
            tile=f'<rect width="64" height="64" rx="14" fill="{tile}"/>',
            c=LIGHT,
        )
    return MARK_SVG.format(label=label, tile="", c=colour)


def draw_mark(size: int, colour: str = TEAL, tile: str | None = None, scale: float = 1.0):
    """The mark as an RGBA Pillow image of ``size`` px, drawn at 4x and downsampled.
    ``tile`` fills a rounded square behind a light mark (the app icon); ``scale`` shrinks the
    mark inside the canvas (Store artwork leaves room for text)."""
    from PIL import Image, ImageDraw

    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fg = LIGHT if tile else colour
    if tile:
        d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 14 / 64), fill=tile)
    u = s / 64 * scale
    off = (s - 64 * u) / 2

    def pt(x: float, y: float) -> tuple[float, float]:
        return (off + x * u, off + y * u)

    def circle(x: float, y: float, r: float) -> None:
        (cx, cy) = pt(x, y)
        d.ellipse((cx - r * u, cy - r * u, cx + r * u, cy + r * u), fill=fg)

    for x, y in SEATS:
        circle(x, y, SEAT_R)
    circle(*CHAIR)
    fx, fy, fw, fh, frx = FRAME
    d.rounded_rectangle(
        (*pt(fx, fy), *pt(fx + fw, fy + fh)),
        radius=int(frx * u),
        outline=fg,
        width=max(1, round(FRAME_STROKE * u)),
    )
    pts = [pt(x, y) for x, y in MOUNTAIN]
    w = max(1, round(MOUNTAIN_STROKE * u))
    d.line(pts, fill=fg, width=w, joint="curve")
    for x, y in pts:  # round caps and joins
        d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fg)
    circle(*SUN)
    return img.resize((size, size), Image.LANCZOS)


def mark_png(size: int, colour: str = TEAL, tile: str | None = None) -> bytes:
    out = io.BytesIO()
    draw_mark(size, colour, tile).save(out, "PNG", optimize=True)
    return out.getvalue()


def _font(size: int, bold: bool):
    from PIL import ImageFont

    candidates = (
        ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    )
    for name in candidates:
        for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")):
            p = folder / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def lockup(width: int, height: int, background: str = TEAL, with_mark: bool = True):
    """Store artwork: the tile colour as background, the light mark, and the lockup
    "Presidium" (medium) + "Image Studio" (regular) beneath it."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), background)
    d = ImageDraw.Draw(img)
    short = min(width, height)
    mark_size = int(short * (0.42 if height >= width else 0.5))
    if with_mark:
        mark = draw_mark(mark_size, LIGHT, None)
        mark_y = (
            int(height * 0.22)
            if height >= width
            else int((height - mark_size) / 2) - int(height * 0.12)
        )
        img.alpha_composite(mark, (int((width - mark_size) / 2), max(0, mark_y)))
    f1 = _font(int(short * 0.085), True)
    f2 = _font(int(short * 0.085), False)
    text_y = (
        (int(height * 0.22) + mark_size + int(short * 0.06))
        if with_mark
        else int(height / 2 - short * 0.05)
    )
    w1 = d.textlength(FAMILY, font=f1)
    w2 = d.textlength(PRODUCT, font=f2)
    gap = int(short * 0.025)
    x = int((width - (w1 + gap + w2)) / 2)
    d.text((x, text_y), FAMILY, font=f1, fill=LIGHT)
    d.text((x + w1 + gap, text_y), PRODUCT, font=f2, fill=LIGHT)
    return img


def write_all(root: Path) -> list[Path]:
    """Regenerate every raster brand asset in the repository from the definition above."""
    from PIL import Image

    written: list[Path] = []

    def save(img, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "PNG", optimize=True)
        written.append(path)

    # Windows icon: every standard size in one .ico
    ico = root / "packaging" / "ImageStudio.ico"
    frames = [draw_mark(s, tile=TEAL).convert("RGBA") for s in (16, 24, 32, 48, 64, 128, 256)]
    frames[-1].save(
        ico, format="ICO", sizes=[(f.width, f.height) for f in frames], append_images=frames[:-1]
    )
    written.append(ico)
    # MSIX assets: keep each file's current size, square ones get the tile, wide ones the lockup
    for p in sorted((root / "packaging" / "msix" / "Assets").glob("*.png")):
        with Image.open(p) as im:
            w, h = im.size
        if w == h:
            save(draw_mark(w, tile=TEAL), p)
        else:
            save(lockup(w, h), p)
    # Store listing artwork
    store = root / "packaging" / "store-assets"
    for name, (w, h) in {
        "box-art-1x1-1080.png": (1080, 1080),
        "box-art-1x1-2160.png": (2160, 2160),
        "poster-9x16-720x1080.png": (720, 1080),
        "poster-9x16-1440x2160.png": (1440, 2160),
        "hero-16x9-1920x1080.png": (1920, 1080),
        "hero-16x9-3840x2160.png": (3840, 2160),
    }.items():
        save(lockup(w, h), store / name)
    for size in (71, 150, 300):
        save(draw_mark(size, tile=TEAL), store / f"display-1x1-{size}.png")
    # the website's icon and the family mark
    (root / "site" / "assets" / "icon.svg").write_text(mark_svg(tile=TEAL), encoding="utf-8")
    written.append(root / "site" / "assets" / "icon.svg")
    return written


if __name__ == "__main__":
    import sys

    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    for p in write_all(repo):
        print(p.relative_to(repo))
