# Building BandSmith

## Windows

Double-click `bandsmith\packaging\build.bat`. It installs what it needs to
build if it isn't there (Python 3.12, the .NET 8 SDK and the WiX 5 toolset,
through winget), makes a private Python environment in
`bandsmith\packaging\.venv` with BandSmith's requirements and the shared
`core` package from this checkout, bundles it with PyInstaller and builds:

- `bandsmith\packaging\dist\bandsmith-0.0.1.msi`: the installer to share
- `bandsmith\packaging\dist\bandsmith\bandsmith.exe`: the app, runnable without installing
- `bandsmith\packaging\dist\bandsmith-setup-0.0.1.exe`: an .exe installer as
  well, only if Inno Setup 6 is installed

Set `BANDSMITH_VERSION=0.0.2` before running it to change the version (an MSI
only upgrades when the number goes up). Releases from GitHub take it from the
tag (`bandsmith-v0.0.2` makes 0.0.2). Versions stay below 1.0 for now.

The installer isn't code-signed, so Windows SmartScreen shows "unknown
publisher" the first time: click "More info", then "Run anyway".

## macOS

On a Mac with Python 3.12, run `sh bandsmith/packaging/build-mac.sh`. It makes
`bandsmith/packaging/dist/BandSmith.app` and
`bandsmith/packaging/dist/bandsmith-<version>-mac.dmg` (Apple Silicon). It
isn't signed with an Apple Developer ID, so the first time it's opened macOS
blocks it: open System Settings > Privacy & Security and click "Open Anyway".

## Not bundled

Stem separation (Demucs and PyTorch, about 1 GB) and lyrics (Whisper) are left
out (`excludes` in bandsmith.spec), as in BeatSmith: they're opt-in downloads
(`rhythmsmith_core.addons`), each showing where it comes from first.

## Icon

TODO: BandSmith needs its own icon. `bandsmith.ico`, `.icns` and `.png` are
BeatSmith's for now.
