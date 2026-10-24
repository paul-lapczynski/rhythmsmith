#!/bin/sh
# Builds the BandSmith app for macOS:
#   dist/BandSmith.app                  the app
#   dist/bandsmith-<version>-mac.dmg    a disk image to share (unsigned, Apple Silicon)
# Needs Python 3.12 (python3.12 on PATH, or set PYTHON). See BUILD.md.
set -e
cd "$(dirname "$0")"
: "${BANDSMITH_VERSION:=0.0.1}"
export BANDSMITH_VERSION

[ -d .venv-mac ] || "${PYTHON:-python3.12}" -m venv .venv-mac
. .venv-mac/bin/activate
python -m pip install --upgrade pip >/dev/null
# the shared core from this checkout, then BandSmith's own requirements
python -m pip install -r ../requirements.txt ../../core pyinstaller

pyinstaller --noconfirm --clean --distpath dist --workpath build bandsmith.spec

rm -rf dmg && mkdir dmg
cp -R dist/BandSmith.app dmg/
ln -s /Applications dmg/Applications
hdiutil create -volname "BandSmith $BANDSMITH_VERSION" -srcfolder dmg -ov -format UDZO "dist/bandsmith-$BANDSMITH_VERSION-mac.dmg"
rm -rf dmg
echo "Done. Disk image: $(pwd)/dist/bandsmith-$BANDSMITH_VERSION-mac.dmg"
