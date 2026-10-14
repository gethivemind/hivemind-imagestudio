"""Entry point of the Windows app (ImageStudio.exe).

Starts the local web UI with its render worker, opens the browser, and keeps a small
console with the address and a log. If a studio is already running on the port (another
window, or the pip-installed `imagestudio ui`), the browser is simply pointed at it.
"""

# Copyright 2026 Ratul Roy (royal-programmer) and the Hivemind contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser

HOST = "127.0.0.1"
PORT = int(os.environ.get("IMAGESTUDIO_PORT", "8766"))


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def open_when_ready(url: str) -> None:
    for _ in range(100):
        if port_open(HOST, PORT):
            break
        time.sleep(0.2)
    webbrowser.open(url)


def main() -> int:
    url = f"http://{HOST}:{PORT}/"
    print("Image Studio by Hivemind")
    print(f"  {url}")
    print("  Everything runs on this machine. Close this window to stop the studio.")
    print()
    if "--version" in sys.argv:
        from imagestudio import __version__

        print(__version__)
        return 0
    if port_open(HOST, PORT):
        print("A studio is already running on that address; opening it in the browser.")
        webbrowser.open(url)
        return 0
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    from imagestudio.ui import serve_ui

    try:
        serve_ui(host=HOST, port=PORT, start_worker=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    # the CLI is also reachable: ImageStudio.exe cli doctor
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from imagestudio.cli import main as cli_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        raise SystemExit(cli_main())
    raise SystemExit(main())
