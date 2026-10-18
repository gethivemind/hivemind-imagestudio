"""Turn this into…: one-click looks for a finished image. Each transform is a tuned
image-to-image pass: wording appended to the picture's own prompt, a negative prompt, and a
strength chosen for the model's measured cliff (on 4-step FLUX, up to about 0.45 keeps the
composition; a medium change like pencil sketch needs about 0.5, a full re-draw like pixel art
about 0.6). The user's saved styles join the list at the default strength."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Transform:
    id: str
    name: str
    text: str  # appended to the source prompt
    negative: str
    strength: float  # how far from the source; see the module docstring
    note: str  # one line for the tile's tooltip


TRANSFORMS: tuple[Transform, ...] = (
    Transform(
        "sketch",
        "Pencil sketch",
        "pencil sketch, graphite on paper, cross-hatching, monochrome",
        "colour, photo, paint",
        0.55,
        "Graphite lines on paper; the composition stays, the colour goes.",
    ),
    Transform(
        "watercolor",
        "Watercolour",
        "watercolour painting, soft washes, paper texture, loose edges",
        "photo, sharp, digital",
        0.5,
        "Soft washes with visible paper; keeps the layout.",
    ),
    Transform(
        "oil",
        "Oil painting",
        "oil painting on canvas, visible brushstrokes, rich colour, gallery lighting",
        "photo, flat, digital",
        0.5,
        "Thick brushwork and deep colour.",
    ),
    Transform(
        "anime",
        "Anime",
        "anime style, clean line art, cel shading, vibrant colours",
        "photo, realistic, 3d render",
        0.55,
        "Clean lines and flat shading.",
    ),
    Transform(
        "pixel",
        "Pixel art",
        "pixel art, 16-bit, limited palette, crisp pixels, no anti-aliasing",
        "photo, smooth, blurry",
        0.6,
        "A retro game sprite of the same scene.",
    ),
    Transform(
        "lineart",
        "Line art",
        "clean black ink line art on white, no shading, minimal",
        "colour, shading, photo",
        0.6,
        "Just the outlines, like a colouring page.",
    ),
    Transform(
        "clay",
        "Claymation",
        "claymation, plasticine figures, stop-motion look, soft studio light",
        "photo, 2d, flat",
        0.55,
        "Plasticine and fingerprints, stop-motion style.",
    ),
    Transform(
        "vintage",
        "Vintage photo",
        "vintage photograph, 1970s film, faded colours, light grain, slight vignette",
        "digital, sharp, modern",
        0.4,
        "Faded film colours; the picture itself hardly changes.",
    ),
    Transform(
        "noir",
        "Film noir",
        "black and white film noir, hard shadows, dramatic low-key lighting, 1940s",
        "colour, bright, flat lighting",
        0.45,
        "Monochrome with hard shadows.",
    ),
    Transform(
        "cyberpunk",
        "Cyberpunk",
        "cyberpunk, neon lights, rain-slick surfaces, night city glow, teal and magenta",
        "daylight, pastel, rustic",
        0.5,
        "Neon and rain over the same scene.",
    ),
    Transform(
        "isometric",
        "Isometric",
        "isometric illustration, clean geometry, pastel palette, soft shadows",
        "photo, perspective, realistic",
        0.6,
        "A tidy isometric diorama.",
    ),
    Transform(
        "storybook",
        "Storybook",
        "children's book illustration, gouache, warm colours, friendly shapes, gentle light",
        "photo, dark, realistic",
        0.55,
        "Warm picture-book illustration.",
    ),
    Transform(
        "product",
        "Product shot",
        "product photography, white seamless background, softbox lighting, sharp focus",
        "clutter, busy background, dim",
        0.45,
        "Clean studio lighting; good for objects.",
    ),
    Transform(
        "3d",
        "3D render",
        "3D render, soft studio lighting, smooth materials, subsurface scattering, high detail",
        "photo, 2d, flat, sketch",
        0.5,
        "A smooth rendered version of the same scene.",
    ),
)

BY_ID: dict[str, Transform] = {t.id: t for t in TRANSFORMS}


def as_dicts() -> list[dict]:
    return [{**asdict(t), "kind": "built-in"} for t in TRANSFORMS]
