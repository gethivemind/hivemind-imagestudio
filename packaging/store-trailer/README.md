# Store trailer

`trailer.py` records the running studio with Playwright into a 1920x1080 video, no sound,
about 50 seconds: six scenes matching the six Store screenshots, short captions drawn by the
page in the Presidium type, and the mark with the lockup on the last frame. It also writes
the 1920x1080 poster frame the Store requires next to a trailer.

    python packaging/store-trailer/trailer.py --url http://127.0.0.1:8766

Outputs in `out/`: `trailer.mp4` (H.264, needs ffmpeg on PATH; otherwise `trailer.webm` is
kept) and `poster.png`. Regenerate per release so the trailer shows the shipped UI.

Reusable: `record(url, out, brand, scenes, end_line)` takes any product's `Brand` (family,
product, colour, mark SVG) and a list of `Scene` objects (caption, action, hold seconds), so
Shape Studio and the desktop app can record their own with the same tool.
