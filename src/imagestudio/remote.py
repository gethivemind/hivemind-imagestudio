"""Phone and remote access: paired devices, one-time pairing codes, request authentication.

The studio normally listens on the loopback address only. With remote access enabled it also
listens on the local network, and every request that does not come from this machine must
carry a device token. Tokens are issued by pairing: the desktop shows a QR code with a
one-time code, the phone opens it, the studio sets a long-lived cookie for that device. Tokens
are stored hashed in ``<data>/devices.json`` and can be revoked from Settings. Nothing ever
leaves the local network unless the user sets up a VPN of their own.
"""

# Copyright 2026 Ratul Roy (royal-programmer) and the Presidium contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import secrets
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

COOKIE = "imagestudio_device"
HEADER = "x-studio-token"
PAIR_TTL_SECONDS = 300
TOKEN_BYTES = 32


def is_loopback(host: str | None) -> bool:
    return host in (None, "", "127.0.0.1", "::1", "localhost", "testclient")


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine on its local networks (best effort, no DNS)."""
    out: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except OSError:
        pass
    # the interface that would carry traffic to the outside is usually the right one
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip not in out:
                out.insert(0, ip)
            elif ip in out:
                out.remove(ip)
                out.insert(0, ip)
    except OSError:
        pass
    return out


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class Device:
    id: str
    name: str
    token_hash: str
    created_at: float
    last_seen: float = 0.0
    user_agent: str = ""

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("token_hash")
        return d


@dataclass
class PairCode:
    code: str
    expires: float
    name: str = "phone"


@dataclass
class Devices:
    """Paired devices on disk plus pending pairing codes in memory."""

    path: Path
    _devices: dict[str, Device] = field(default_factory=dict)
    _pending: dict[str, PairCode] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._devices = {d["id"]: Device(**d) for d in raw.get("devices", [])}
        except (OSError, ValueError, TypeError, KeyError):
            self._devices = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"devices": [asdict(d) for d in self._devices.values()]}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # -- pairing ------------------------------------------------------------------------------

    def start_pairing(self, name: str = "phone") -> PairCode:
        with self._lock:
            now = time.time()
            self._pending = {c: p for c, p in self._pending.items() if p.expires > now}
            code = secrets.token_urlsafe(9)
            pc = PairCode(code=code, expires=now + PAIR_TTL_SECONDS, name=name)
            self._pending[code] = pc
            return pc

    def complete_pairing(self, code: str, user_agent: str = "") -> tuple[Device, str] | None:
        """Exchange a one-time code for a device and its token (returned once, never stored)."""
        with self._lock:
            pc = self._pending.pop(code, None)
            if pc is None or pc.expires < time.time():
                return None
            token = secrets.token_urlsafe(TOKEN_BYTES)
            dev = Device(
                id=secrets.token_hex(6),
                name=pc.name,
                token_hash=_hash(token),
                created_at=time.time(),
                last_seen=time.time(),
                user_agent=user_agent[:200],
            )
            self._devices[dev.id] = dev
            self._save()
            return dev, token

    def pending_count(self) -> int:
        now = time.time()
        return sum(1 for p in self._pending.values() if p.expires > now)

    # -- authentication -----------------------------------------------------------------------

    def authenticate(self, token: str | None) -> Device | None:
        if not token:
            return None
        h = _hash(token)
        with self._lock:
            for d in self._devices.values():
                if secrets.compare_digest(d.token_hash, h):
                    if time.time() - d.last_seen > 60:
                        d.last_seen = time.time()
                        self._save()
                    return d
        return None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [d.public() for d in sorted(self._devices.values(), key=lambda d: d.created_at)]

    def rename(self, device_id: str, name: str) -> bool:
        with self._lock:
            d = self._devices.get(device_id)
            if d is None:
                return False
            d.name = name.strip()[:60] or d.name
            self._save()
            return True

    def revoke(self, device_id: str) -> bool:
        with self._lock:
            if self._devices.pop(device_id, None) is None:
                return False
            self._save()
            return True

    def revoke_all(self) -> int:
        with self._lock:
            n = len(self._devices)
            self._devices = {}
            self._save()
            return n


def qr_svg(payload: str) -> str:
    """An SVG QR code for ``payload`` (segno, pure Python)."""
    import io

    import segno

    buf = io.BytesIO()
    segno.make(payload, error="m").save(
        buf, kind="svg", scale=6, border=2, dark="#1b1b19", light=None
    )
    return buf.getvalue().decode("utf-8")
