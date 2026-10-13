"""Which model files are installed and verified. A small JSON manifest avoids re-hashing
tens of gigabytes on every start: a file counts as installed when it exists with the pinned
size and the manifest records its hash as verified."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imagestudio import download as dl
from imagestudio.config import Settings
from imagestudio.registry import (
    MODELS,
    TOOL_MODELS,
    VISION_MODELS,
    FileSpec,
    ModelSpec,
    ToolModel,
    Variant,
    VisionModel,
)

ConsentFn = Callable[[str, int], bool]  # (description, bytes) -> allowed?
ProgressFn = Callable[[str, int, int], None]  # (filename, done, total)


def _size_matches(path, size: int) -> bool:
    """Catches truncation after verification. Tests patch this instead of writing 10 GB."""
    return path.stat().st_size == size


@dataclass
class InstalledVariant:
    model: str
    variant: str
    bytes: int


class ModelStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._manifest: dict[str, str] = {}
        self._load()

    # -- manifest ------------------------------------------------------------------------

    def _load(self) -> None:
        p = self.settings.manifest_file
        if p.exists():
            try:
                self._manifest = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._manifest = {}

    def _save(self) -> None:
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.manifest_file.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    # -- queries -------------------------------------------------------------------------

    def file_installed(self, spec: FileSpec) -> bool:
        if spec.local_path:
            return Path(spec.local_path).is_file()
        p = self.settings.models_dir / spec.filename
        return (
            p.exists()
            and _size_matches(p, spec.size)
            and self._manifest.get(spec.filename) == spec.sha256
        )

    def variant_installed(self, variant: Variant) -> bool:
        return all(self.file_installed(f) for f in variant.files)

    def installed_variants(self, model: ModelSpec) -> list[Variant]:
        return [v for v in model.ranked_variants if self.variant_installed(v)]

    def installed(self) -> list[InstalledVariant]:
        out: list[InstalledVariant] = []
        for m in MODELS.values():
            out.extend(
                InstalledVariant(m.name, v.name, v.download_bytes)
                for v in self.installed_variants(m)
            )
        return out

    def missing_bytes(self, variant: Variant) -> int:
        return sum(f.size for f in variant.files if not self.file_installed(f))

    def path_for(self, spec: FileSpec):
        if spec.local_path:
            return Path(spec.local_path)
        return self.settings.models_dir / spec.filename

    # -- install -------------------------------------------------------------------------

    def install_variant(
        self,
        model: ModelSpec,
        variant: Variant,
        consent: ConsentFn,
        progress: ProgressFn | None = None,
    ) -> bool:
        """Download every missing file of ``variant`` after ``consent`` approves the total.

        Returns False (and downloads nothing) when consent is refused.
        """
        missing = [f for f in variant.files if not self.file_installed(f)]
        if not missing:
            return True
        total = sum(f.size for f in missing)
        desc = f"{model.display_name} [{variant.name}] ({model.licence}): " + ", ".join(
            f.filename for f in missing
        )
        if not consent(desc, total):
            return False
        for f in missing:
            target = self.path_for(f)

            def _p(done: int, tot: int, name: str = f.filename) -> None:
                if progress:
                    progress(name, done, tot)

            dl.download(f.url, target, f.sha256, f.size, progress=_p)
            self._manifest[f.filename] = f.sha256
            self._save()
        return True

    def tool_installed(self, tool: ToolModel) -> bool:
        return self.file_installed(tool.file)

    def installed_tools(self) -> list[str]:
        return [t.name for t in TOOL_MODELS.values() if self.tool_installed(t)]

    def install_tool(
        self, tool: ToolModel, consent: ConsentFn, progress: ProgressFn | None = None
    ) -> bool:
        if self.tool_installed(tool):
            return True
        if not consent(
            f"{tool.display_name} ({tool.licence}): {tool.file.filename}", tool.file.size
        ):
            return False
        target = self.path_for(tool.file)

        def _p(done: int, tot: int, name: str = tool.file.filename) -> None:
            if progress:
                progress(name, done, tot)

        dl.download(tool.file.url, target, tool.file.sha256, tool.file.size, progress=_p)
        self._manifest[tool.file.filename] = tool.file.sha256
        self._save()
        return True

    def vision_installed(self, vm: VisionModel) -> bool:
        return all(self.file_installed(f) for f in vm.files)

    def installed_vision(self) -> list[str]:
        return [v.name for v in VISION_MODELS.values() if self.vision_installed(v)]

    def install_vision(
        self, vm: VisionModel, consent: ConsentFn, progress: ProgressFn | None = None
    ) -> bool:
        missing = [f for f in vm.files if not self.file_installed(f)]
        if not missing:
            return True
        total = sum(f.size for f in missing)
        if not consent(
            f"{vm.display_name} ({vm.licence}): " + ", ".join(f.filename for f in missing), total
        ):
            return False
        for f in missing:

            def _p(done: int, tot: int, name: str = f.filename) -> None:
                if progress:
                    progress(name, done, tot)

            dl.download(f.url, self.path_for(f), f.sha256, f.size, progress=_p)
            self._manifest[f.filename] = f.sha256
            self._save()
        return True

    def remove_files(self, files) -> int:
        """Delete the given files and forget their hashes. Returns bytes freed."""
        freed = 0
        for f in files:
            p = self.path_for(f)
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            part = p.with_suffix(p.suffix + ".part")
            if part.exists():
                part.unlink()
            self._manifest.pop(f.filename, None)
        self._save()
        return freed

    def file_info(self, spec: FileSpec) -> dict:
        p = self.path_for(spec)
        return {
            "filename": spec.filename,
            "path": str(p),
            "installed": self.file_installed(spec),
            "bytes": p.stat().st_size if p.exists() else 0,
            "size": spec.size,
            "licence": spec.licence,
            "source": spec.source,
        }

    def verify_file(self, spec: FileSpec) -> bool:
        """Re-hash a file on disk and update the manifest. Slow; used by ``doctor --verify``."""
        p = self.path_for(spec)
        if not p.exists():
            return False
        ok = dl.sha256_of(p) == spec.sha256
        if ok:
            self._manifest[spec.filename] = spec.sha256
        else:
            self._manifest.pop(spec.filename, None)
        self._save()
        return ok
