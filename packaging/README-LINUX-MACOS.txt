Image Studio by Hivemind - macOS and Linux builds
==================================================

macOS (Apple silicon)
---------------------
1. Unzip, drag "Image Studio.app" to Applications (or run it from anywhere).
2. First start: the app is not notarized yet, so macOS says it "cannot be opened because the
   developer cannot be verified". Right-click (or Control-click) the app, choose Open, then
   Open again. macOS remembers the choice. Or: System Settings > Privacy & Security > Open Anyway.
   Check the download against the .sha256 file on the release page if you want to be sure it
   is the official build.
3. The studio opens in a window of its own (the system WebKit view). There is no tray icon on
   macOS: closing the window during a render asks whether to stop it. "Image Studio.app
   --browser" uses your browser instead.
   Logs: ~/Library/Application Support/hivemind-imagestudio/logs/app.log
4. The image engine (the official stable-diffusion.cpp macOS build) and the models are downloaded by the
   studio on first use, hash-verified, after your consent.

Linux (x86_64, glibc 2.39 or newer: Ubuntu 24.04, Fedora 40, Debian 13 and later)
-------------------------------------------------------------------------------
1. Unpack:   tar -xzf ImageStudio-<version>-linux-x86_64.tar.gz
2. Run:      ./ImageStudio-<version>/ImageStudio
   The studio starts its local server and opens in your default browser; Ctrl+C stops it.
   A native window would need GTK or Qt that this build cannot carry, so the browser is the
   honest choice here (the same interface, same quality). --no-browser runs the server only.
   Logs: ~/.local/share/hivemind-imagestudio/logs/app.log
3. The image engine (stable-diffusion.cpp, CPU and Vulkan builds for Ubuntu 24.04) and the
   models are downloaded by the studio on first use, hash-verified, after your consent.
   Vulkan needs the vulkan-loader package of your distribution (libvulkan1).

Everything runs on this machine. Nothing is uploaded.

Command line (both)
-------------------
./ImageStudio cli doctor
./ImageStudio cli generate "a lighthouse at dusk" --preset best

Prefer Python?  pipx install hivemind-imagestudio   then   imagestudio ui
