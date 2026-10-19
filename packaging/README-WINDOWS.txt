Presidium Image Studio - Windows app
======================================

1. Unzip this folder anywhere (for example C:\Apps\ImageStudio).
2. Run ImageStudio.exe. The studio opens in a window of its own. Closing it during a render
   offers to keep rendering in the background (tray icon; click it to come back). Starting it
   again focuses the running window. ImageStudio.exe --browser uses your browser instead.
   Logs: %LOCALAPPDATA%\hivemind-imagestudio\logs\app.log
3. First time: Home shows a getting-started checklist. Install the engine and download a
   model from there (or from Settings). Downloads are hash-verified and need your consent.

Everything runs on this machine. Nothing is uploaded.

Windows may warn
----------------
This build is not code-signed yet, so Windows SmartScreen shows "Windows protected your PC"
the first time you run a downloaded copy. Click "More info", then "Run anyway". It asks once
per downloaded file. Check the download against the .sha256 file on the release page if you
want to be sure it is the official build.

Endpoint protection (Trend Micro, Defender and similar) may also ask about the image engine
(sd-cli.exe) and the vision engine (llama-mtmd-cli.exe) the first time each runs: they are
downloaded by the studio from the pinned stable-diffusion.cpp and llama.cpp releases and
their hashes are in the source. Allow them, or add the tools folder under
%LOCALAPPDATA%\hivemind-imagestudio	ools as a trusted path, and it stops asking.

Command line
------------
ImageStudio.exe cli doctor
ImageStudio.exe cli generate "a lighthouse at dusk" --preset best
ImageStudio.exe cli --help

Licence: Apache-2.0 (LICENSE, NOTICE). The name and logo are trademarks (TRADEMARK.md).
Made with love by Ratul Roy (royal-programmer). https://presidiumdev.github.io/presidium-imagestudio/
