# PyInstaller spec for the BandSmith app. Built by build.bat on Windows and
# build-mac.sh on macOS; see BUILD.md.
# TODO: BandSmith needs its own icon; bandsmith.ico/.icns/.png are BeatSmith's for now.
import os
import sys

app_dir = SPECPATH + "/.."                 # bandsmith/
core_dir = SPECPATH + "/../../core"        # the shared rhythmsmith_core package
version = os.environ.get("BANDSMITH_VERSION", "0.0.1").lstrip("v")
# the app's version, for what it tells web services about itself (rhythmsmith_core.artwork)
with open(SPECPATH + "/version.txt", "w") as f:
    f.write(version)
datas = [(SPECPATH + "/version.txt", "."), (SPECPATH + "/bandsmith.png", "."),
         # run by the optional add-ons' own Python, so they ship as plain files too
         (core_dir + "/rhythmsmith_core/stems.py", "rhythmsmith_core"),
         (core_dir + "/rhythmsmith_core/lyrics.py", "rhythmsmith_core"),
         (core_dir + "/rhythmsmith_core/accel.py", "rhythmsmith_core"),
         (core_dir + "/rhythmsmith_core/beats.py", "rhythmsmith_core")]
hidden = ["rhythmsmith_core", "rhythmsmith_core.accel", "rhythmsmith_core.addons",
          "rhythmsmith_core.artwork", "rhythmsmith_core.lyrics", "rhythmsmith_core.stems",
          "rhythmsmith_core.beats", "mutagen", "mido", "mido.backends"]
binaries = []
from PyInstaller.utils.hooks import collect_all
for pkg in ("librosa", "soundfile", "lazy_loader", "audioread", "soxr", "mido"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

a = Analysis([SPECPATH + "/run_bandsmith.py"], pathex=[app_dir, core_dir], datas=datas, binaries=binaries,
             hiddenimports=hidden, excludes=["tkinter", "matplotlib", "IPython",
                      "torch", "torchaudio", "demucs"])   # stem separation is a separate download
pyz = PYZ(a.pure)
mac = sys.platform == "darwin"
icon = SPECPATH + ("/bandsmith.icns" if mac else "/bandsmith.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="bandsmith", console=False,
          icon=icon if os.path.isfile(icon) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="bandsmith")
if mac:
    app = BUNDLE(coll, name="BandSmith.app", icon=icon, bundle_identifier="io.github.paul-lapczynski.bandsmith",
                 version=version,
                 info_plist={"CFBundleDisplayName": "BandSmith", "NSHighResolutionCapable": True,
                             "LSMinimumSystemVersion": "12.0"})
