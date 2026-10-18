"""Face detection with UltraFace (MIT) on onnxruntime, and the crop-and-paste helpers behind
the "fix faces" tool. The detector finds faces; the fix itself is ordinary image-to-image on
each face crop at a higher resolution, pasted back under a feathered mask. Everything runs on
this machine; the model is 1.5 MB."""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from imagestudio.engine.base import EngineError

Box = tuple[int, int, int, int]  # x, y, w, h in source pixels


def _ensure_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise EngineError(
            "face detection needs onnxruntime, numpy and Pillow: "
            "pip install 'hivemind-imagestudio[bg]'"
        ) from e
    return np, ort, Image


class FaceDetector:
    """UltraFace RFB-640: input 1x3x480x640 RGB, (x-127)/128; outputs scores 1xNx2 and boxes
    1xNx4 as fractions of the image. Stateless apart from a cached session per model."""

    name = "onnxruntime"

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}

    def available(self) -> bool:
        try:
            _ensure_runtime()
        except EngineError:
            return False
        return True

    def _session(self, model_path: Path):
        key = str(model_path)
        if key not in self._sessions:
            _np, ort, _img = _ensure_runtime()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # the exported graph trips harmless initializer warnings
            self._sessions[key] = ort.InferenceSession(
                key, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def detect(self, image: Path, model_path: Path, threshold: float = 0.7) -> list[Box]:
        """Faces as (x, y, w, h) boxes in source pixels, largest first."""
        np, _ort, image_mod = _ensure_runtime()
        sess = self._session(model_path)
        inp = sess.get_inputs()[0]
        shape = inp.shape  # [1, 3, H, W]
        ih = int(shape[2]) if isinstance(shape[2], int) else 480
        iw = int(shape[3]) if isinstance(shape[3], int) else 640
        with image_mod.open(image) as src:
            im = src.convert("RGB")
            w, h = im.size
            small = im.resize((iw, ih), image_mod.BILINEAR)
        x = (np.asarray(small, dtype=np.float32) - 127.0) / 128.0
        x = x.transpose(2, 0, 1)[None]
        scores, boxes = sess.run(None, {inp.name: x})[:2]
        conf = scores[0][:, 1]
        keep = conf > threshold
        cand = [(float(c), b) for c, b in zip(conf[keep], boxes[0][keep], strict=True)]
        cand.sort(key=lambda t: -t[0])
        picked: list[tuple[float, float, float, float]] = []
        for _c, b in cand:
            bx = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            if all(_iou(bx, p) < 0.3 for p in picked):
                picked.append(bx)
        out: list[Box] = []
        for x0, y0, x1, y1 in picked:
            px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
            px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
            if px1 - px0 >= 8 and py1 - py0 >= 8:
                out.append((px0, py0, px1 - px0, py1 - py0))
        out.sort(key=lambda b: -(b[2] * b[3]))
        return out


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def expand_box(box: Box, width: int, height: int, factor: float = 1.8) -> Box:
    """A square region around the face with room for hair and chin, clamped to the image."""
    x, y, w, h = box
    side = int(max(w, h) * factor)
    side = min(side, width, height)
    cx, cy = x + w // 2, y + h // 2
    x0 = min(max(0, cx - side // 2), width - side)
    y0 = min(max(0, cy - side // 2), height - side)
    return (x0, y0, side, side)


def crop_for_render(image: Path, region: Box, size: int, out: Path) -> None:
    """Cut the region out and scale it to size x size for the engine."""
    from PIL import Image

    x, y, w, h = region
    with Image.open(image) as im:
        crop = im.convert("RGB").crop((x, y, x + w, y + h)).resize((size, size), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out, "PNG")


def paste_back(base, rendered: Path, region: Box, feather: float = 0.12):
    """Scale the rendered crop back to the region and blend it in under an elliptical,
    feathered mask so the seam disappears. ``base`` is a PIL image, modified in place."""
    from PIL import Image, ImageDraw, ImageFilter

    x, y, w, h = region
    with Image.open(rendered) as r:
        patch = r.convert("RGB").resize((w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    inset = int(min(w, h) * feather)
    ImageDraw.Draw(mask).ellipse((inset, inset, w - inset, h - inset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, inset)))
    base.paste(patch, (x, y), mask)
    return base
