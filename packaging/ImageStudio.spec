# PyInstaller spec for the Windows app. Build from the repository root:
#   pyinstaller packaging/ImageStudio.spec --noconfirm
# Produces dist/ImageStudio/ (one folder; zip it for distribution). A one-file build was
# rejected on purpose: it unpacks itself into a temp folder on every start, which is slow
# and looks more suspicious to endpoint protection than a plain folder.

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("starlette")
    + collect_submodules("mcp", filter=lambda n: not n.startswith("mcp.cli"))
    + collect_submodules("imagestudio")
    + collect_submodules("webview")
    + collect_submodules("pystray")
    + ["truststore", "clr", "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin", "PIL.IcoImagePlugin"]
)
datas = [(str(SRC / "imagestudio" / "static"), "imagestudio/static")]
datas += collect_data_files("truststore")
datas += collect_data_files("mcp")
datas += collect_data_files("webview")
datas += [(str(ROOT / "packaging" / "ImageStudio.ico"), ".")]
try:
    import onnxruntime  # noqa: F401

    hidden += collect_submodules("onnxruntime")
    datas += collect_data_files("onnxruntime")
except ImportError:
    pass

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest", "playwright", "mcp.cli", "typer"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # a real window; logs go to <data>/logs/app.log
    icon=str(ROOT / "packaging" / "ImageStudio.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ImageStudio",
)
