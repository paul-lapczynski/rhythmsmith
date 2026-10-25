"""Compare BandSmith's Expert charts with reference charts (local evaluation).

    python bandsmith/tools/eval_vs_reference.py SONGS_DIR --out REPORTS_DIR [--limit 20] [--seed 0]

Each folder in SONGS_DIR is a song with a reference notes.mid and its audio:
mix.ogg (the full mix) if there is one, else every .ogg in it mixed together;
sep/ (stems already separated, vocals.ogg ...) is used when present. For each
song this runs `make` on the full mix into REPORTS_DIR/charts/<folder>/, runs
scan-chart on the result when Node and tools/scan-chart are installed, and
compares Expert with the reference:

    onset F1        notes on the same 16th-note slot of the reference's tempo map
                    (f1), or within one slot (f1_near)
    chord rate      share of notes with 2+ lanes, ours and the reference's
    nps             notes per second, ours and the reference's
    direction       of matched consecutive notes, how often ours moves the same
                    way (up, down, same) as the reference (guitar and bass)
    drum lanes      F1 per drum lane (kick, snare, yellow, blue, green)

Writes REPORTS_DIR/eval.csv (one row per song and instrument) and eval.md (totals).
Keep REPORTS_DIR out of git: rows name songs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "core"))

from bandsmith.chart import read_mid  # noqa: E402
from bandsmith.chart.model import TempoMap  # noqa: E402

SLOT = 120                  # a 16th note at 480 ticks per beat
PARTS = ("guitar", "bass", "drums", "vocals")


def slots(times: np.ndarray, ref_tm: TempoMap) -> np.ndarray:
    return np.array([int(round(ref_tm.tick(t) / SLOT)) for t in times], int)


def f1(pred: np.ndarray, true: np.ndarray, tol: int = 0) -> tuple[float, float, float]:
    if len(pred) == 0 or len(true) == 0:
        return 0.0, 0.0, 0.0
    true = np.sort(true)
    used = np.zeros(len(true), bool)
    hit = 0
    for p in np.sort(pred):
        i = np.searchsorted(true, p - tol)
        while i < len(true) and true[i] <= p + tol:
            if not used[i]:
                used[i] = True
                hit += 1
                break
            i += 1
    pr, rc = hit / len(pred), hit / len(true)
    return pr, rc, (2 * pr * rc / (pr + rc) if hit else 0.0)


def notes_of(song, part: str):
    """[(seconds, lanes)] for a part's Expert notes (vocals: pitch as a 1-lane tuple)."""
    tm = song.tempo_map()
    if part == "vocals":
        return [(tm.seconds(n.tick), (n.pitch or 0,)) for n in (song.vocals.notes if song.vocals else [])]
    p = song.parts.get(part)
    return [(tm.seconds(n.tick), n.lanes) for n in (p.notes.get("expert", []) if p else [])]


def compare(ours, ref, pad: float = 0.0) -> dict:
    """ours/ref: Song objects; pad: seconds of silence `make` put in front of ours."""
    rtm = ref.tempo_map()
    out = {}
    for part in PARTS:
        a, b = notes_of(ours, part), notes_of(ref, part)
        if not b:
            continue
        at = np.array([t - pad for t, _ in a])
        bt = np.array([t for t, _ in b])
        sa, sb = slots(at, rtm), slots(bt, rtm)
        _, _, exact = f1(sa, sb)
        pr, rc, near = f1(sa, sb, 1)
        dur = max(1.0, bt[-1] - bt[0]) if len(bt) > 1 else 1.0
        row = {"f1": exact, "f1_near": near, "precision_near": pr, "recall_near": rc,
               "nps": len(a) / dur, "ref_nps": len(b) / dur,
               "chord_rate": float(np.mean([len(l) > 1 for _, l in a])) if a else 0.0,
               "ref_chord_rate": float(np.mean([len(l) > 1 for _, l in b]))}
        if part in ("guitar", "bass") and a:
            # direction agreement on matched notes (same slot)
            ref_by = {s: np.mean(l) for s, (_, l) in zip(sb, b)}
            pairs = [(np.mean(l), ref_by[s]) for s, (_, l) in zip(sa, a) if s in ref_by]
            agree = [np.sign(x2 - x1) == np.sign(y2 - y1) for (x1, y1), (x2, y2) in zip(pairs, pairs[1:])]
            row["direction"] = float(np.mean(agree)) if agree else 0.0
            # the same frets (the whole chord) within a 16th
            row["fret_f1_near"] = fret_f1(sa, a, sb, b)
            lanes_by = {s: l for s, (_, l) in zip(sb, b)}
            sizes = [len(l) == len(lanes_by[s]) for s, (_, l) in zip(sa, a) if s in lanes_by]
            row["chord_size_agree"] = float(np.mean(sizes)) if sizes else 0.0
        if part == "drums":
            for lane, name in enumerate(("kick", "snare", "yellow", "blue", "green")):
                la = slots(np.array([t - pad for t, l in a if lane in l]), rtm)
                lb = slots(np.array([t for t, l in b if lane in l]), rtm)
                row[f"{name}_f1_near"] = f1(la, lb, 1)[2]
            # tom or cymbal, on the yellow/blue/green hits that line up
            ca = {(s, lane): lane in c for s, (_, l), c in zip(sa, a, cymbals_of(ours)) for lane in l if lane >= 2}
            cb = {(s, lane): lane in c for s, (_, l), c in zip(sb, b, cymbals_of(ref)) for lane in l if lane >= 2}
            both = [ca[k] == cb[k] for k in ca if k in cb]
            row["cymbal_agree"] = float(np.mean(both)) if both else 0.0
        out[part] = row
    return out


def fret_f1(sa, a, sb, b) -> float:
    """F1 of notes that land within a 16th of a reference note with exactly the same frets."""
    if not len(sa) or not len(sb):
        return 0.0
    ka = [tuple(l) for _, l in a]
    kb = [tuple(l) for _, l in b]
    hits = 0
    for key in set(ka) & set(kb):
        pa = np.array([s for s, k in zip(sa, ka) if k == key])
        pb = np.array([s for s, k in zip(sb, kb) if k == key])
        pr, _, _ = f1(pa, pb, 1)
        hits += round(pr * len(pa))
    p, r = hits / len(sa), hits / len(sb)
    return 2 * p * r / (p + r) if hits else 0.0


def cymbals_of(song) -> list:
    p = song.parts.get("drums")
    return [n.cymbals for n in (p.notes.get("expert", []) if p else [])]


def scan(folder: Path) -> dict | None:
    import shutil
    node = shutil.which("node")
    check = HERE / "scan-chart" / "check.mjs"
    if node is None or not (HERE / "scan-chart" / "node_modules" / "scan-chart").is_dir():
        return None
    r = subprocess.run([node, str(check), str(folder)], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None


def full_mix(song_dir: Path, tmp: Path) -> Path:
    if (song_dir / "mix.ogg").is_file():
        return song_dir / "mix.ogg"
    from bandsmith.analyze import audio
    ys = [audio.load(p, audio.OUT_SR, mono=False) for p in sorted(song_dir.glob("*.ogg"))
          if p.stem not in ("preview", "crowd")]
    n = max(len(y) for y in ys)
    y = np.zeros((n, 2), np.float32)
    for part in ys:
        y[: len(part)] += part
    y *= min(1.0, 0.98 / (np.abs(y).max() or 1.0))
    out = tmp / f"{song_dir.name}.mix.ogg"
    audio.write_ogg(out, y)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("songs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", help="a file listing which song folders to use, one per line")
    ap.add_argument("--no-lyrics", action="store_true")
    a = ap.parse_args(argv)
    from bandsmith.analyze.make import make
    dirs = sorted(d for d in a.songs.iterdir() if (d / "notes.mid").is_file())
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        dirs = [d for d in dirs if d.name in keep]
    else:
        random.Random(a.seed).shuffle(dirs)
    dirs = dirs[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, scans = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, d in enumerate(dirs):
            t = time.time()
            mix = full_mix(d, Path(tmp))
            sep = d / "sep" if (d / "sep").is_dir() else None
            res = make(mix, a.out / "charts" / d.name, stems_dir=sep, whisper=not a.no_lyrics,
                       title=d.name, artist="", log=lambda s: None)
            secs = time.time() - t
            ref = read_mid(d / "notes.mid")
            cmp = compare(res.song, ref, res.pad)
            sc = scan(res.folder)
            issues = sorted({x["noteIssue"] for x in sc["chartIssues"]}) if sc else ["(not scanned)"]
            folder_issues = sorted({x["folderIssue"] for x in sc["folderIssues"]}) if sc else []
            scans.append({"song": d.name, "playable": sc["playable"] if sc else None,
                          "chart_issues": issues, "folder_issues": folder_issues})
            for part, row in cmp.items():
                rows.append({"song": d.name, "part": part, "seconds": round(secs, 1), **{k: round(v, 3) for k, v in row.items()}})
            print(f"{i + 1}/{len(dirs)} {d.name}: {secs:.0f} s, "
                  + ", ".join(f"{p} {r['f1_near']:.2f}" for p, r in cmp.items()), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("song", "part", "seconds"), k))
    with open(a.out / "eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (a.out / "scan.json").write_text(json.dumps(scans, indent=1))
    lines = ["# BandSmith vs reference (Expert)", "", f"- songs: {len(dirs)}",
             f"- seconds per song: median {np.median([r['seconds'] for r in rows if r['part'] == rows[0]['part']]):.0f}"
             if rows else "", ""]
    lines += ["| part | songs | f1 | f1 near | precision | recall | nps (ref) | chord rate (ref) | direction |",
              "|---|---|---|---|---|---|---|---|---|"]
    for part in PARTS:
        rs = [r for r in rows if r["part"] == part]
        if not rs:
            continue
        m = lambda k: np.mean([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")  # noqa: E731
        lines.append(f"| {part} | {len(rs)} | {m('f1'):.3f} | {m('f1_near'):.3f} | {m('precision_near'):.3f} | "
                     f"{m('recall_near'):.3f} | {m('nps'):.2f} ({m('ref_nps'):.2f}) | "
                     f"{m('chord_rate'):.2f} ({m('ref_chord_rate'):.2f}) | {m('direction'):.2f} |")
    drum = [r for r in rows if r["part"] == "drums"]
    if drum:
        lines += ["", "Drum lanes (F1 within a 16th): " + ", ".join(
            f"{n} {np.mean([r[f'{n}_f1_near'] for r in drum]):.3f}" for n in ("kick", "snare", "yellow", "blue", "green"))]
    playable = sum(1 for s in scans if s["playable"])
    clean = sum(1 for s in scans if s["chart_issues"] == [])
    from collections import Counter
    iss = Counter(i for s in scans for i in s["chart_issues"])
    fi = Counter(i for s in scans for i in s["folder_issues"])
    lines += ["", f"scan-chart: {playable}/{len(scans)} playable, {clean}/{len(scans)} with no chart issues",
              f"- chart issues (songs): {dict(iss)}", f"- folder issues (songs): {dict(fi)}"]
    (a.out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
