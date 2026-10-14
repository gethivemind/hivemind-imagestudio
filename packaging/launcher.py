"""ImageStudio.exe: the studio as a Windows application.

The local server and its render worker run inside this process; the interface is shown in a
native window (WebView2, which ships with Windows 10 and 11). Closing the window while a job
renders offers to keep rendering in the background with a tray icon. A second launch focuses
the existing window instead of starting another studio.

Options:  --browser  (use the default browser instead of a window)
          --no-browser (headless server only)
          --version
          cli …  (run the command line, e.g. ImageStudio.exe cli doctor)
"""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("IMAGESTUDIO_PORT", "8766"))
CONTROL_PORT = PORT + 1  # single-instance handshake
TITLE = "Image Studio by Hivemind"
WEBVIEW2_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((HOST, port)) == 0


# ---------------------------------------------------------------- logging to a file (no console)


def setup_logging(data_dir: Path) -> Path:
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "app.log"
    handlers: list[logging.Handler] = [logging.FileHandler(path, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    # a windowed exe has no console; keep print() from crashing on None streams
    if sys.stdout is None:
        sys.stdout = open(path, "a", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = sys.stdout
    return path


# ---------------------------------------------------------------- the server in a thread


class Server:
    def __init__(self) -> None:
        self.studio = None
        self.server = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        from imagestudio.service import Studio
        from imagestudio.ui import build_app

        self.studio = Studio(start_worker=True)
        config = uvicorn.Config(
            build_app(self.studio), host=HOST, port=PORT, log_level="warning", lifespan="off"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="ui-server", daemon=True)
        self.thread.start()
        for _ in range(150):
            if port_open(PORT):
                return
            time.sleep(0.1)
        raise RuntimeError("the studio server did not start")

    def rendering(self) -> bool:
        if self.studio is None:
            return False
        return any(j.status in ("queued", "running") for j in self.studio.jobs.all())

    def progress_text(self) -> str:
        if self.studio is None:
            return TITLE
        running = [j for j in self.studio.jobs.all() if j.status == "running"]
        queued = [j for j in self.studio.jobs.all() if j.status == "queued"]
        if running:
            j = running[0]
            pct = int((j.progress or 0) * 100) if hasattr(j, "progress") else 0
            more = f" · {len(queued)} queued" if queued else ""
            return f"{TITLE}\nrendering {j.step}/{j.total_steps or '?'} ({pct}%){more}"
        if queued:
            return f"{TITLE}\n{len(queued)} queued"
        return f"{TITLE}\nidle"

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(10)
        if self.studio is not None:
            self.studio.close()


# ---------------------------------------------------------------- single instance


def claim_instance(on_focus) -> socket.socket | None:
    """Bind the control port; return the socket, or None when another instance holds it
    (which was then asked to come to the front)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, CONTROL_PORT))
    except OSError:
        s.close()
        try:
            with socket.create_connection((HOST, CONTROL_PORT), timeout=1) as c:
                c.sendall(b"focus\n")
        except OSError:
            pass
        return None
    s.listen(2)

    def serve() -> None:
        while True:
            try:
                conn, _ = s.accept()
            except OSError:
                return
            with conn, contextlib.suppress(OSError):
                conn.recv(16)
            on_focus()

    threading.Thread(target=serve, name="control", daemon=True).start()
    return s


# ---------------------------------------------------------------- the window


class App:
    def __init__(self, server: Server, data_dir: Path, log_path: Path) -> None:
        self.server = server
        self.data_dir = data_dir
        self.log_path = log_path
        self.window = None
        self.tray = None
        self.hidden = False
        self.quitting = False
        self.state_file = data_dir / "window.json"

    # ---- js api, reachable as window.pywebview.api.* from the UI
    def pick_files(self, kind: str = "image", multiple: bool = False):
        import webview

        types = {
            "image": ("Images (*.png;*.jpg;*.jpeg;*.webp)",),
            "lora": ("LoRA weights (*.safetensors;*.gguf;*.ckpt;*.pt)",),
            "weights": ("Model weights (*.gguf;*.safetensors;*.ckpt)",),
        }.get(kind, ("All files (*.*)",))
        res = self.window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=multiple, file_types=types
        )
        return list(res or [])

    def app_info(self):
        return {"windowed": True, "log": str(self.log_path), "data_dir": str(self.data_dir)}

    def open_in_browser(self):
        webbrowser.open(f"http://{HOST}:{PORT}/")
        return True

    # ---- window state
    def load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_state(self) -> None:
        w = self.window
        if w is None:
            return
        try:
            self.state_file.write_text(
                json.dumps({"width": w.width, "height": w.height, "x": w.x, "y": w.y}),
                encoding="utf-8",
            )
        except Exception:  # best effort
            logging.getLogger("app").exception("window state not saved")

    # ---- show / hide / focus
    def focus(self) -> None:
        if self.window is None:
            return
        if self.hidden:
            self.window.show()
            self.hidden = False
        with contextlib.suppress(Exception):
            self.window.restore()

    def on_closing(self) -> bool:
        """Return False to cancel the close (we hide to the tray instead)."""
        if self.quitting:
            self.save_state()
            return True
        if self.server.rendering():
            keep = self.window.create_confirmation_dialog(
                TITLE,
                "A render is still running.\n\nKeep rendering in the background? The studio stays "
                "in the tray; click its icon to come back. Choose No to stop the render and quit.",
            )
            if keep:
                self.save_state()
                self.hide_to_tray()
                return False
        self.quitting = True
        self.save_state()
        self.stop_tray()
        return True

    def hide_to_tray(self) -> None:
        self.hidden = True
        self.window.hide()
        self.start_tray()

    # ---- tray (only while hidden)
    def start_tray(self) -> None:
        if self.tray is not None:
            return
        try:
            import pystray
            from PIL import Image

            icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "ImageStudio.ico"
            image = (
                Image.open(icon_path)
                if icon_path.exists()
                else Image.new("RGB", (64, 64), "#f0b429")
            )
            menu = pystray.Menu(
                pystray.MenuItem("Open Image Studio", lambda: self.focus(), default=True),
                pystray.MenuItem("Open in browser", lambda: self.open_in_browser()),
                pystray.MenuItem("Quit", lambda: self.quit()),
            )
            self.tray = pystray.Icon("imagestudio", image, self.server.progress_text(), menu)
            threading.Thread(target=self.tray.run, name="tray", daemon=True).start()
            threading.Thread(target=self._tray_tooltip, name="tray-tooltip", daemon=True).start()
        except Exception:
            logging.getLogger("app").exception("tray icon unavailable")

    def _tray_tooltip(self) -> None:
        while self.tray is not None:
            with contextlib.suppress(Exception):
                self.tray.title = self.server.progress_text()
            time.sleep(3)

    def stop_tray(self) -> None:
        if self.tray is not None:
            with contextlib.suppress(Exception):
                self.tray.stop()
            self.tray = None

    def quit(self) -> None:
        self.quitting = True
        self.stop_tray()
        if self.window is not None:
            with contextlib.suppress(Exception):
                self.window.destroy()

    # ---- run
    def run(self) -> None:
        import webview

        st = self.load_state()
        self.window = webview.create_window(
            TITLE,
            f"http://{HOST}:{PORT}/",
            width=int(st.get("width", 1400)),
            height=int(st.get("height", 900)),
            x=st.get("x"),
            y=st.get("y"),
            min_size=(960, 640),
            js_api=self,
            text_select=True,
        )
        self.window.events.closing += self.on_closing
        storage = self.data_dir / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        webview.start(gui="edgechromium", private_mode=False, storage_path=str(storage))


# ---------------------------------------------------------------- WebView2 presence (Windows)


def webview2_available() -> bool:
    if sys.platform != "win32":
        return True
    import winreg

    keys = (
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ),
        (
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        ),
    )
    for root, key in keys:
        try:
            with winreg.OpenKey(root, key) as k:
                pv, _ = winreg.QueryValueEx(k, "pv")
                if pv and pv != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


def ask_webview2() -> bool:
    """Offer the WebView2 download; True = use the browser for this run instead."""
    if sys.platform != "win32":
        return True
    import ctypes

    r = ctypes.windll.user32.MessageBoxW(
        0,
        "Image Studio shows its window with Microsoft WebView2, which is missing on this "
        "computer.\n\nYes: open the download page (about 100 MB, from Microsoft) and use the "
        "browser for now.\nNo: use the browser for now.\nCancel: quit.",
        TITLE,
        0x03 | 0x40,  # Yes/No/Cancel, information icon
    )
    if r == 6:  # yes
        webbrowser.open(WEBVIEW2_URL)
        return True
    return r == 7  # no: browser for this run; cancel: quit


# ---------------------------------------------------------------- main


def main() -> int:
    if "--version" in sys.argv:
        from imagestudio import __version__

        print(__version__)
        return 0
    from imagestudio.config import Settings

    settings = Settings()
    log_path = setup_logging(settings.data_dir)
    log = logging.getLogger("app")
    url = f"http://{HOST}:{PORT}/"

    if port_open(PORT):
        # a studio already serves this port: another window (focus it) or `imagestudio ui`
        if port_open(CONTROL_PORT):
            try:
                with socket.create_connection((HOST, CONTROL_PORT), timeout=1) as c:
                    c.sendall(b"focus\n")
                return 0
            except OSError:
                pass
        webbrowser.open(url)
        return 0

    server = Server()
    server.start()
    log.info("studio at %s (worker in this process: %s)", url, server.studio.worker_active)

    if "--no-browser" in sys.argv:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        server.stop()
        return 0

    use_browser = "--browser" in sys.argv
    if not use_browser and not webview2_available():
        if not ask_webview2():
            server.stop()
            return 0
        use_browser = True
    if use_browser:
        webbrowser.open(url)
        print(f"{TITLE}: {url}  (press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        server.stop()
        return 0

    app = App(server, settings.data_dir, log_path)
    control = claim_instance(app.focus)
    try:
        app.run()  # blocks until the window is destroyed
    finally:
        if control is not None:
            control.close()
        app.stop_tray()
        server.stop()
        log.info("stopped")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from imagestudio.cli import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        raise SystemExit(cli_main())
    raise SystemExit(main())
