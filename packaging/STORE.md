# Installer and the Microsoft Store

Two installable forms of the Windows app come out of the release workflow:

| File | Signed by | For |
| --- | --- | --- |
| `ImageStudio-<ver>-Setup.exe` | nobody (yet) | the normal installer: Start menu, search, Pin to Start, uninstall; SmartScreen asks once |
| `ImageStudio-<ver>-windows-x64.zip` | nobody (yet) | portable: unzip and run; SmartScreen asks once |
| `ImageStudio-<ver>-windows-x64-store.msix` | Microsoft, on submission | upload to Partner Center |

A third, `…-sideload.msix` with a throw-away certificate, is only for local testing
(`python packaging/build_msix.py --version X --sideload`).

## Why the Store

An MSIX from the Store is signed by Microsoft, so there is no SmartScreen warning, no
certificate to buy or renew, and updates are delivered by the Store. Developer registration is
free for individuals since September 2025 (and for companies since May 2026); if the sign-up ever
shows a fee, stop and re-check, the official blog says there is none. `winget install Hivemind.ImageStudio` becomes
possible through the Store as well. The app keeps running fully offline; the Store only
delivers the package.

## Status

Listed since 2026-10-16: Store ID **9NJ8PVHV74KZ**, https://apps.microsoft.com/detail/9NJ8PVHV74KZ,
`winget install 9NJ8PVHV74KZ -s msstore`. First certified submission: 0.3.6.

## Steps, once

1. Register at https://developer.microsoft.com/microsoft-store/register (individual account; free). Developer name: **Hivemind** (or your own name with
   Hivemind as the publisher display name).
2. Create the app with the name **Image Studio by Hivemind**. Reserving the name is the
   trademark-relevant step: do it early.
3. Open *Product management → Product identity* and copy the three values:
   `Package/Identity/Name` (looks like `12345Hivemind.ImageStudio`),
   `Package/Identity/Publisher` (`CN=…GUID…`), `Package/Properties/PublisherDisplayName`.
4. In the GitHub repository, add them as Actions **variables** (not secrets, they are public
   in the manifest anyway): `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER`, `MSIX_PUBLISHER_DISPLAY`.
   From then on every release builds the Store package with the right identity.

## Steps, per release

1. Download `ImageStudio-<ver>-windows-x64-store.msix` from the GitHub Release.
2. Partner Center → the app → *Start update* → *Packages* → upload the `.msix`.
3. Store listing: screenshots are in `site/assets/shot-*.png`; the description is the hero
   text on the website; category *Photo & video*; age rating via the questionnaire.
4. Submit. Certification takes one to three days the first time, usually hours afterwards.

## Known requirements the package already meets

- `runFullTrust` capability (a PyInstaller app is a classic desktop app), declared.
- The exe is referenced as `ImageStudio\ImageStudio.exe`; the PyInstaller folder is copied in
  whole.
- Tile and logo assets in `packaging/msix/Assets` are generated from the brand mark by
  the script that produced them (see git history) at every required scale.
- The package writes only under `%LOCALAPPDATA%\hivemind-imagestudio`, which MSIX allows.

## Things to decide later

- **Winget directly** (without the Store) also works with the zip plus a manifest in the
  `winget-pkgs` repository, but then SmartScreen still applies to the unsigned exe.
- **Code-signing certificate** for the zip: only worth it if many users prefer the zip over
  the Store; Azure Trusted Signing (about US$10/month) is the cheapest route if eligible.
