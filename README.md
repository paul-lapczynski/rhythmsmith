# RhythmSmith

A monorepo for rhythm-game chart tools.

- **BandSmith**: a Clone Hero / Rock Band chart generator. In progress
  (0.0.1: the app window only, nothing is generated yet).
- **BeatSmith**: the StepMania (DDR) chart generator. It lives at
  [paul-lapczynski/beatsmith](https://github.com/paul-lapczynski/beatsmith) for now.

## Layout

```
core/                     rhythmsmith_core: shared by the apps
  rhythmsmith_core/
    stems.py              Demucs stem separation (htdemucs; htdemucs_6s adds guitar and piano)
    accel.py              which processor neural networks run on (CUDA, Apple MPS, CPU)
    addons.py             opt-in downloads, each showing its source URL first
    lyrics.py             Whisper: the words off a vocals stem
    artwork.py            a song's cover art (beside it, in its tags, or online when asked)
    beats.py              Beat This! beat tracking (an opt-in add-on)
    notes.py              basic-pitch note detection (an opt-in add-on)
  pyproject.toml
bandsmith/
  bandsmith/              the app (PySide6)
    chart/                the song model; notes.mid (read and write), song.ini, tempo map from beats
    analyze/              stems, beats, transcription, the music layer (music.py), charting, the AI model
  tests/                  tests (no song files needed: they make their own)
  tools/                  demo_song.py (a synth song with an exact chart, for tests), eval_vs_reference.py,
                          scan-chart/ (test-only chart checker, Node), inventory.py
  packaging/              PyInstaller spec, Windows MSI/.exe and macOS .dmg builds (BUILD.md)
  requirements.txt
docs/                     the future GitHub Pages site
.github/workflows/        bandsmith.yml: builds Windows and macOS for changes to bandsmith/ or core/
```

## Run BandSmith from source

Python 3.12 (3.10 or newer works):

```bash
python -m venv .venv
.venv/Scripts/activate        # macOS/Linux: . .venv/bin/activate
pip install -r bandsmith/requirements.txt -e core pytest
cd bandsmith
python -m bandsmith           # opens the BandSmith window
python -m bandsmith make "Some Song.mp3" -o "out/Some Song"   # an Expert Clone Hero song folder
python -m pytest tests        # tests; the scan-chart ones need Node:
npm ci --prefix tools/scan-chart
```

To build the installers, see [bandsmith/packaging/BUILD.md](bandsmith/packaging/BUILD.md).

Songs, audio, artwork and charts are never committed (see `.gitignore`).

MIT licensed; see [LICENSE](LICENSE).
