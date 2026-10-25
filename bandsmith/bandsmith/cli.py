"""The command line:

    python -m bandsmith make SONG -o OUTDIR [--parts guitar,bass,drums,vocals]
                             [--art-online] [--ai-beats] [--no-ai-notes] [--no-lyrics]
                             [--stems DIR] [--cpu] [--ai-model]

OUTDIR becomes a Clone Hero / YARG song folder (notes.mid, song.ini, the stems
as .ogg, album.png). --art-online looks the cover up online (each address is
printed first); --ai-beats uses Beat This! if its add-on is installed; --ai-model
places guitar, bass and drums with the AI model when its files are in its folder. Add-ons
are installed from the app, or with python -m rhythmsmith_core.addons install KIND.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bandsmith", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="chart a song into a Clone Hero song folder")
    m.add_argument("song", type=Path)
    m.add_argument("-o", "--out", type=Path, required=True)
    m.add_argument("--parts", default="guitar,bass,drums,vocals")
    m.add_argument("--art-online", action="store_true", help="look the album art up online")
    m.add_argument("--ai-beats", action="store_true", help="beats from Beat This! (add-on)")
    m.add_argument("--no-ai-notes", action="store_true", help="don't use basic-pitch even if installed")
    m.add_argument("--no-lyrics", action="store_true", help="don't read lyrics with Whisper")
    m.add_argument("--stems", type=Path, help="a folder of stems already separated (vocals.ogg, ...)")
    m.add_argument("--cache", type=Path, help="where separated stems are kept")
    m.add_argument("--title")
    m.add_argument("--artist")
    m.add_argument("--cpu", action="store_true", help="don't use the GPU")
    m.add_argument("--ai-model", action="store_true", help="use the AI model where it's installed")
    a = ap.parse_args(argv)
    if a.cpu:
        os.environ["STEPGEN_DEVICE"] = "cpu"
    from .analyze.make import make
    parts = tuple(p.strip() for p in a.parts.split(",") if p.strip())
    r = make(a.song, a.out, parts, stems_dir=a.stems, cache=a.cache, art_online=a.art_online,
             ai_beats=a.ai_beats, ai_notes=not a.no_ai_notes, whisper=not a.no_lyrics,
             title=a.title, artist=a.artist, log=lambda s: print(s, file=sys.stderr, flush=True),
             ai_model=a.ai_model)
    print(f"{r.folder}: " + ", ".join(f"{k} {v} notes" for k, v in r.notes.items())
          + f" in {r.timings['total']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
