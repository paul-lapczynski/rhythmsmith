"""Inventory a folder of rhythm-game song charts (Clone Hero / Rock Band style).

    python bandsmith/tools/inventory.py <folder> [--out <folder>/reports] [--7z PATH]

Reads, never changes, what's in the folder: loose song folders, .zip archives
(nested .zip ones too, in memory), .7z archives (with 7-Zip: the listing, then
only the small chart and song.ini files are unpacked into a temporary folder
that is deleted afterwards), .sng packages and Rock Band CON/LIVE files
(counted; their insides aren't read). Nested .7z/.rar archives are counted only.

For each song it records the chart format (notes.mid, notes.chart, .sng), song.ini,
the audio layout (one mixed song file or separate stems) and formats, which
instruments and difficulties the chart has, pro drums, 2x kick, open notes,
taps, the charter (from song.ini), the song length, likely duplicates and
charters that look like auto-generators. It writes:

    <out>/inventory.csv     one row per song
    <out>/summary.md        the totals

Keep <out> inside the (git-ignored) data folder: the rows name songs.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

AUDIO_EXT = {"ogg", "opus", "mp3", "mogg", "wav", "flac", "m4a"}
CHART_FILES = {"notes.mid", "notes.chart"}
SMALL = CHART_FILES | {"song.ini"}
NESTED_EXT = {"zip", "7z", "rar"}
# stem names Clone Hero / YARG read: song, guitar, bass, rhythm, keys, drums(_1-4),
# vocals(_1-2), crowd, preview
STEM_RE = re.compile(r"^(song|guitar|bass|rhythm|keys|drums(_\d)?|vocals(_\d)?|crowd|preview)$")
AUTO_RE = re.compile(r"\b(auto\w*|ai|a\.i\.|generat\w*|bot|machine|neural|ml)\b", re.I)
DIFFS = ("Easy", "Medium", "Hard", "Expert")
DIFF_LETTER = {"Easy": "E", "Medium": "M", "Hard": "H", "Expert": "X"}

# MIDI track names -> instrument
MIDI_PARTS = {
    "PART GUITAR": "guitar", "T1 GEMS": "guitar", "PART BASS": "bass", "PART RHYTHM": "rhythm",
    "PART GUITAR COOP": "guitar_coop", "PART KEYS": "keys", "PART DRUMS": "drums",
    "PART DRUMS_2X": "drums_2x", "PART VOCALS": "vocals", "HARM1": "harm1", "HARM2": "harm2",
    "HARM3": "harm3", "PART REAL_GUITAR": "pro_guitar", "PART REAL_GUITAR_22": "pro_guitar",
    "PART REAL_BASS": "pro_bass", "PART REAL_BASS_22": "pro_bass", "PART REAL_KEYS_X": "pro_keys",
    "PART GHL GUITAR": "ghl_guitar", "PART GUITAR GHL": "ghl_guitar", "PART BASS GHL": "ghl_bass",
}
# .chart sections: [<Difficulty><Instrument>]
CHART_INSTR = {"Single": "guitar", "DoubleGuitar": "guitar_coop", "DoubleBass": "bass",
               "DoubleRhythm": "rhythm", "Keyboard": "keys", "Drums": "drums",
               "GHLGuitar": "ghl_guitar", "GHLBass": "ghl_bass"}
REPORT_PARTS = ("guitar", "bass", "rhythm", "keys", "drums", "vocals", "harm1", "pro_guitar",
                "pro_bass", "pro_keys", "ghl_guitar")


# ------------------------------------------------------------------ sources

@dataclass
class Entry:
    path: str                       # posix path inside its source
    size: int
    read: object = None             # () -> bytes, for the small files
    head: object = None             # () -> the first bytes, for telling CON/LIVE files apart


@dataclass
class Song:
    source: str
    folder: str
    files: dict = field(default_factory=dict)        # lower-case name -> Entry
    kind: str = "folder"                              # folder | sng | con


def zip_entries(zf: zipfile.ZipFile, label: str, out: list, problems: list, depth=0):
    """Entries of a zip; nested zips are opened in memory (seekable member)."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        ext = name.rsplit(".", 1)[-1].lower() if "." in name.rsplit("/", 1)[-1] else ""
        if ext == "zip" and depth < 3:
            try:
                with zf.open(info) as f:
                    inner = zipfile.ZipFile(f if f.seekable() else io.BytesIO(f.read()))
                    zip_entries(inner, f"{label}!{name}", out, problems, depth + 1)
                continue
            except (zipfile.BadZipFile, OSError, RuntimeError) as e:
                problems.append(f"bad nested zip in {label}: {name} ({e})")
        out.append((label, Entry(name, info.file_size, (lambda zf=zf, info=info: zf.read(info)),
                                 (lambda zf=zf, info=info: zf.open(info).read(4)))))


def sevenzip_entries(exe: str, path: str, label: str, out: list, problems: list, tmp: str):
    r = subprocess.run([exe, "l", "-slt", "-ba", "-sccUTF-8", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        problems.append(f"7-Zip couldn't list {label}: {r.stderr.strip()[:200]}")
        return
    items, cur = [], {}
    for line in r.stdout.splitlines() + [""]:
        if not line.strip():
            if "Path" in cur:
                items.append(cur)
            cur = {}
        elif " = " in line:
            k, v = line.split(" = ", 1)
            cur[k] = v
    wanted = []
    for it in items:
        if "D" in it.get("Attributes", "") or it.get("Folder") == "+":
            continue
        name = it["Path"].replace("\\", "/")
        base = name.rsplit("/", 1)[-1].lower()
        out.append((label, Entry(name, int(it.get("Size") or 0))))
        if base in SMALL:
            wanted.append(out[-1])
    if not wanted:
        return
    # one pass over the (often solid) archive for just the small files
    dest = os.path.join(tmp, str(len(os.listdir(tmp))))
    args = [exe, "x", "-y", "-bso0", "-bsp0", f"-o{dest}", path] + [f"-ir!{n}" for n in sorted(SMALL)]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        problems.append(f"7-Zip couldn't unpack the chart files of {label}: {r.stderr.strip()[:200]}")
    for _, e in wanted:
        p = long_path(os.path.join(dest, *e.path.split("/")))
        e.read = (lambda p=p: open(p, "rb").read())


def long_path(p: str) -> str:
    """Windows paths past 260 characters need the \\\\?\\ prefix."""
    return "\\\\?\\" + os.path.abspath(p) if os.name == "nt" and not p.startswith("\\\\?\\") else p


def folder_entries(root: str, label: str, out: list):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            out.append((label, Entry(rel, os.path.getsize(full), (lambda p=full: open(p, "rb").read()),
                                     (lambda p=full: open(p, "rb").read(4)))))


# ------------------------------------------------------------------ charts

def read_vlq(b: bytes, i: int) -> tuple[int, int]:
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i


def parse_midi(data: bytes) -> tuple[dict, float]:
    """{track name: {"notes": Counter(note), "texts": set, "sysex": count}}, length in seconds."""
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hlen, = struct.unpack(">I", data[4:8])
    _, ntrk, div = struct.unpack(">HHH", data[8:14])
    i = 8 + hlen
    tracks, tempos, last_tick = {}, [], 0
    for t in range(ntrk):
        if data[i:i + 4] != b"MTrk":
            break
        tlen, = struct.unpack(">I", data[i + 4:i + 8])
        b, j, end = data, i + 8, i + 8 + tlen
        i = end
        tick, status, name = 0, 0, None
        notes, texts, sysex, last_note = Counter(), set(), 0, 0
        while j < end:
            d, j = read_vlq(b, j)
            tick += d
            c = b[j]
            if c == 0xFF:
                typ = b[j + 1]
                ln, j = read_vlq(b, j + 2)
                body = b[j:j + ln]
                j += ln
                if typ == 0x03 and name is None:
                    name = body.decode("latin-1").strip()
                elif typ in (0x01, 0x05):
                    txt = body.decode("latin-1").strip()
                    if len(texts) < 400:
                        texts.add(txt)
                elif typ == 0x51 and ln == 3:
                    tempos.append((tick, int.from_bytes(body, "big")))
                elif typ == 0x2F:
                    break
                continue
            if c in (0xF0, 0xF7):
                ln, j = read_vlq(b, j + 1)
                body = b[j:j + ln]
                j += ln
                if body[:3] == b"PS\x00":
                    sysex += 1
                continue
            if c & 0x80:
                status = c
                j += 1
            hi = status & 0xF0
            if hi in (0xC0, 0xD0):
                j += 1
            else:
                a1, a2 = b[j], b[j + 1]
                j += 2
                if hi == 0x90 and a2 > 0:
                    notes[a1] += 1
                    last_note = max(last_note, tick)
        last_tick = max(last_tick, last_note)
        tracks[name or f"track{t}"] = {"notes": notes, "texts": texts, "sysex": sysex}
    return tracks, ticks_to_sec(last_tick, sorted(tempos), div)


def ticks_to_sec(tick: int, tempos: list, res: int) -> float:
    sec, prev, uspq = 0.0, 0, 500000
    for t, us in tempos:
        if t >= tick:
            break
        sec += (t - prev) * uspq / res / 1e6
        prev, uspq = t, us
    return sec + (tick - prev) * uspq / res / 1e6


def midi_facts(data: bytes) -> dict:
    tracks, length = parse_midi(data)
    parts, flags = {}, set()
    for name, tr in tracks.items():
        part = MIDI_PARTS.get(name.upper())
        if not part:
            continue
        n = tr["notes"]
        if part in ("vocals", "harm1", "harm2", "harm3"):
            if sum(c for k, c in n.items() if 36 <= k <= 84):
                parts[part] = ["Expert"]          # vocals have one chart
            continue
        if part.startswith("pro_"):
            base = {"pro_keys": None}.get(part, 0)
            have = [d for d, lo in zip(DIFFS, (24, 48, 72, 96)) if sum(n[k] for k in range(lo, lo + 6))]
            if base is None:
                have = ["Expert"] if n else []
            if have:
                parts[part] = have
            continue
        lo_hi = [(60, 64), (72, 76), (84, 88), (96, 100)]
        if part.startswith("ghl"):
            lo_hi = [(58, 64), (70, 76), (82, 88), (94, 100)]
        have = [d for d, (lo, hi) in zip(DIFFS, lo_hi) if sum(n[k] for k in range(lo, hi + 1))]
        if have:
            parts[part] = have
        if part in ("drums", "drums_2x"):
            if n[95]:
                flags.add("double_kick")
            if any(n[k] for k in (110, 111, 112)):
                flags.add("pro_drums")
        else:
            if n[104]:
                flags.add("taps")
            if "[ENHANCED_OPENS]" in tr["texts"] or "ENHANCED_OPENS" in tr["texts"]:
                flags.add("open_notes")
            if tr["sysex"]:
                flags.add("sysex_open_or_tap")
    return {"parts": parts, "flags": flags, "length": length}


def chart_facts(data: bytes) -> dict:
    text = data.decode("utf-8-sig", errors="replace")
    sections, cur = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1]
            sections[cur] = []
        elif cur and "=" in s:
            sections[cur].append(s)
    res = 192
    for l in sections.get("Song", []):
        k, v = (x.strip() for x in l.split("=", 1))
        if k == "Resolution":
            res = int(re.sub(r"\D", "", v) or 192)
    tempos = []
    for l in sections.get("SyncTrack", []):
        k, v = (x.strip() for x in l.split("=", 1))
        bits = v.split()
        if bits[0] == "B":
            tempos.append((int(k), round(60e6 / (int(bits[1]) / 1000))))
    parts, flags, last = {}, set(), 0
    for sec, lines in sections.items():
        m = re.fullmatch(r"(Easy|Medium|Hard|Expert)(\w+)", sec)
        if not m or m.group(2) not in CHART_INSTR:
            continue
        notes = Counter()
        for l in lines:
            k, v = (x.strip() for x in l.split("=", 1))
            bits = v.split()
            if bits and bits[0] == "N":
                notes[int(bits[1])] += 1
                last = max(last, int(k))
        if not notes:
            continue
        part = CHART_INSTR[m.group(2)]
        parts.setdefault(part, []).append(m.group(1))
        if part == "drums":
            if notes[32]:
                flags.add("double_kick")
            if any(notes[k] for k in (66, 67, 68)):
                flags.add("pro_drums")
        else:
            if notes[7]:
                flags.add("open_notes")
            if notes[6]:
                flags.add("taps")
    parts = {p: [d for d in DIFFS if d in ds] for p, ds in parts.items()}
    return {"parts": parts, "flags": flags, "length": ticks_to_sec(last, sorted(tempos), res)}


def read_ini(data: bytes) -> dict:
    text = data.decode("utf-8-sig", errors="replace")
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read_string(text if text.lstrip().startswith("[") else "[song]\n" + text)
    except configparser.Error:
        return {}
    for s in cp.sections():
        if s.strip().lower() == "song":
            return {k.lower(): v.strip() for k, v in cp[s].items()}
    return {}


def sng_meta(data: bytes) -> tuple[dict, dict]:
    """(metadata, {file name: bytes}) of a .sng package (only small files decoded)."""
    if data[:6] != b"SNGPKG":
        raise ValueError("not an SNG file")
    mask = data[10:26]
    i = 26
    _, count = struct.unpack("<QQ", data[i:i + 16])
    i += 16
    meta = {}
    for _ in range(count):
        kl, = struct.unpack("<i", data[i:i + 4]); i += 4
        k = data[i:i + kl].decode("utf-8", "replace"); i += kl
        vl, = struct.unpack("<i", data[i:i + 4]); i += 4
        meta[k.lower()] = data[i:i + vl].decode("utf-8", "replace"); i += vl
    _, count = struct.unpack("<QQ", data[i:i + 16])
    i += 16
    files = {}
    for _ in range(count):
        nl = data[i]; i += 1
        name = data[i:i + nl].decode("utf-8", "replace"); i += nl
        ln, off = struct.unpack("<QQ", data[i:i + 16]); i += 16
        if name.lower() in CHART_FILES:
            raw = data[off:off + ln]
            files[name.lower()] = bytes(c ^ mask[k % 16] ^ (k & 0xFF) for k, c in enumerate(raw))
        else:
            files[name.lower()] = ln
    return meta, files


# ------------------------------------------------------------------ songs

def norm(s: str) -> str:
    s = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", "", s.casefold())
    s = re.sub(r"\b(feat|ft)\..*$", "", s)
    return re.sub(r"[^\w]+", "", s)


def group_songs(entries: list) -> tuple[list, list]:
    songs: dict[tuple, Song] = {}
    others = []
    for label, e in entries:
        p = PurePosixPath(e.path)
        base = p.name.lower()
        ext = base.rsplit(".", 1)[-1] if "." in base else ""
        if ext == "sng":
            songs[(label, e.path)] = Song(label, e.path, {base: e}, "sng")
            continue
        if ext in NESTED_EXT:
            others.append(("nested_" + ext, label, e))
            continue
        if not ext and e.size > 1 << 20:
            others.append(("maybe_con", label, e))
            continue
        key = (label, str(p.parent))
        songs.setdefault(key, Song(label, str(p.parent))).files[base] = e
    keep = [s for s in songs.values()
            if s.kind == "sng" or any(f in s.files for f in SMALL)
            or any(f.rsplit(".", 1)[-1] in AUDIO_EXT for f in s.files)]
    return keep, others


def con_magic(e: Entry) -> str | None:
    if e.head is None:
        return None
    try:
        head = e.head()
    except Exception:
        return None
    return {b"CON ": "CON", b"LIVE": "LIVE", b"PIRS": "PIRS"}.get(head)


def describe(song: Song, problems: list) -> dict:
    row = {"source": song.source, "folder": song.folder, "kind": song.kind}
    files = dict(song.files)
    ini = {}
    chart_data = {}
    if song.kind == "sng":
        try:
            meta, inner = sng_meta(files[next(iter(files))].read())
            ini = meta
            chart_data = {k: v for k, v in inner.items() if isinstance(v, bytes)}
            files = {k: Entry(k, v if isinstance(v, int) else len(v)) for k, v in inner.items()}
        except Exception as e:
            problems.append(f"bad .sng {song.source}:{song.folder} ({e})")
    else:
        for name in CHART_FILES:
            if name in files and files[name].read:
                try:
                    chart_data[name] = files[name].read()
                except Exception as e:
                    problems.append(f"unreadable {name} in {song.source}:{song.folder} ({e})")
        if "song.ini" in files and files["song.ini"].read:
            try:
                ini = read_ini(files["song.ini"].read())
            except Exception as e:
                problems.append(f"unreadable song.ini in {song.source}:{song.folder} ({e})")
    has_mid, has_chart = "notes.mid" in files, "notes.chart" in files
    row["format"] = ("sng" if song.kind == "sng" else "mid+chart" if has_mid and has_chart
                     else "mid" if has_mid else "chart" if has_chart else "none")
    row["song_ini"] = song.kind == "sng" or "song.ini" in song.files
    audio = sorted(f for f in files if f.rsplit(".", 1)[-1] in AUDIO_EXT)
    stems = sorted({f.rsplit(".", 1)[0] for f in audio if STEM_RE.match(f.rsplit(".", 1)[0])}
                   - {"preview", "crowd"})
    row["audio_formats"] = " ".join(sorted({f.rsplit(".", 1)[-1] for f in audio}))
    row["audio_stems"] = " ".join(stems)
    instrument_stems = [s for s in stems if s != "song"]
    row["audio_layout"] = ("none" if not audio else "mogg" if any(f.endswith(".mogg") for f in audio)
                           else "stems" if instrument_stems else "single")
    facts = {"parts": {}, "flags": set(), "length": 0.0}
    src = "notes.mid" if "notes.mid" in chart_data else "notes.chart" if "notes.chart" in chart_data else None
    if src:
        try:
            facts = (midi_facts if src == "notes.mid" else chart_facts)(chart_data[src])
        except Exception as e:
            problems.append(f"unparsable {src} in {song.source}:{song.folder} ({type(e).__name__}: {e})")
    for p in REPORT_PARTS:
        row[p] = "".join(DIFF_LETTER[d] for d in facts["parts"].get(p, []))      # e.g. "EMHX"
    row["harmonies"] = sum(1 for h in ("harm1", "harm2", "harm3") if h in facts["parts"])
    pro = str(ini.get("pro_drums", "")).lower() in ("true", "1")
    row["pro_drums"] = pro or "pro_drums" in facts["flags"]
    row["double_kick"] = "double_kick" in facts["flags"] or "drums_2x" in facts["parts"]
    row["open_notes"] = "open_notes" in facts["flags"]
    row["taps"] = "taps" in facts["flags"]
    row["sysex_markers"] = "sysex_open_or_tap" in facts["flags"]
    row["artist"] = ini.get("artist", "")
    row["title"] = ini.get("name", "")
    if not row["title"]:
        leaf = PurePosixPath(song.folder).name
        if " - " in leaf:
            row["artist"], row["title"] = (x.strip() for x in leaf.split(" - ", 1))
        else:
            row["title"] = leaf
    row["charter"] = ini.get("charter") or ini.get("frets", "")
    ms = re.sub(r"\D", "", str(ini.get("song_length", "")))
    row["length_s"] = round(int(ms) / 1000, 1) if ms else round(facts["length"], 1)
    row["auto_charter_hint"] = bool(row["charter"] and AUTO_RE.search(row["charter"]))
    row["dup_key"] = norm(row["artist"]) + "|" + norm(row["title"])
    missing = []
    if row["format"] == "none":
        missing.append("no chart")
    if row["audio_layout"] == "none":
        missing.append("no audio")
    row["problems"] = "; ".join(missing)
    return row


# ------------------------------------------------------------------ report

def summary(rows: list, others: list, sources: list, problems: list) -> str:
    n = len(rows)
    dup = Counter(r["dup_key"] for r in rows if r["dup_key"] != "|")
    out = ["# Chart inventory", ""]
    total = sum(s for _, s in sources)
    out += [f"- Sources: {len(sources)} ({total / 1e9:.2f} GB)",
            f"- Songs (folders or packages with a chart or audio): {n}",
            f"- Unique songs (artist + title, ignoring bracketed parts): {len(dup)}",
            f"- Duplicate groups: {sum(1 for c in dup.values() if c > 1)} "
            f"({sum(c - 1 for c in dup.values() if c > 1)} extra copies)", ""]
    kinds = Counter(k for k, _, _ in others)
    out += ["## Not read", ""] + [f"- {k}: {c}" for k, c in sorted(kinds.items())] + \
           (["- none"] if not kinds else []) + [""]
    out += ["## Chart format", ""] + [f"- {k}: {c}" for k, c in Counter(r["format"] for r in rows).most_common()] + [""]
    out += ["## Audio", ""] + [f"- {k}: {c}" for k, c in Counter(r["audio_layout"] for r in rows).most_common()]
    fmts = Counter(f for r in rows for f in r["audio_formats"].split())
    stems = Counter(s for r in rows for s in r["audio_stems"].split())
    out += [f"- formats (songs using each): {', '.join(f'{k} {c}' for k, c in fmts.most_common())}",
            f"- stem files (songs with each): {', '.join(f'{k} {c}' for k, c in stems.most_common())}", ""]
    out += ["## Instruments", "", "| part | charted | with Expert | all four difficulties |", "|---|---|---|---|"]
    for p in REPORT_PARTS:
        have = [r[p] for r in rows if r[p]]
        if have:
            out.append(f"| {p} | {len(have)} | {sum('X' in h for h in have)} | {sum(h == 'EMHX' for h in have)} |")
    out += ["", f"- vocal harmonies (2+ parts): {sum(r['harmonies'] >= 2 for r in rows)}",
            f"- pro drums: {sum(r['pro_drums'] for r in rows)}",
            f"- 2x kick: {sum(r['double_kick'] for r in rows)}",
            f"- open notes: {sum(r['open_notes'] for r in rows)}",
            f"- taps: {sum(r['taps'] for r in rows)}",
            f"- Phase Shift sysex markers (open/tap): {sum(r['sysex_markers'] for r in rows)}",
            f"- song.ini present: {sum(r['song_ini'] for r in rows)}", ""]
    lens = sorted(r["length_s"] for r in rows if r["length_s"])
    if lens:
        out += [f"- length: {len(lens)} known, median {lens[len(lens) // 2] / 60:.1f} min, "
                f"total {sum(lens) / 3600:.1f} h", ""]
    charters = Counter(r["charter"] for r in rows if r["charter"])
    out += ["## Charters", "", f"- distinct: {len(charters)}; no charter given: {sum(not r['charter'] for r in rows)}",
            f"- looks auto-generated: {sum(r['auto_charter_hint'] for r in rows)}", ""]
    bad = Counter(r["problems"] for r in rows if r["problems"])
    out += ["## Problems", ""] + [f"- {k}: {c}" for k, c in bad.items()] + \
           [f"- {p}" for p in problems] + (["- none"] if not bad and not problems else [])
    return "\n".join(out) + "\n"


def find_7z(given: str | None) -> str | None:
    for c in (given, shutil.which("7z"), shutil.which("7za"), r"C:\Program Files\7-Zip\7z.exe",
              r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if c and os.path.isfile(c):
            return c
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", help="where the reports go (default: <folder>/reports)")
    ap.add_argument("--7z", dest="sevenzip", help="7-Zip's 7z executable, for .7z archives")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.folder)
    out_dir = os.path.abspath(args.out or os.path.join(root, "reports"))
    sevenzip = find_7z(args.sevenzip)
    entries, problems, sources, open_zips = [], [], [], []
    tmp = tempfile.mkdtemp(prefix="inventory-")
    try:
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if full == out_dir:
                continue
            low = name.lower()
            if os.path.isdir(full):
                folder_entries(full, name, entries)
                continue
            sources.append((name, os.path.getsize(full)))
            print(f"reading {name}", file=sys.stderr, flush=True)
            if low.endswith(".zip"):
                try:
                    zf = zipfile.ZipFile(full)
                    open_zips.append(zf)
                    zip_entries(zf, name, entries, problems)
                except (zipfile.BadZipFile, OSError) as e:
                    problems.append(f"bad zip {name} ({e})")
            elif low.endswith(".7z"):
                if sevenzip:
                    sevenzip_entries(sevenzip, full, name, entries, problems, tmp)
                else:
                    problems.append(f"{name}: no 7-Zip found (--7z), not read")
            elif low.endswith(".sng"):
                entries.append((".", Entry(name, os.path.getsize(full), (lambda p=full: open(p, "rb").read()),
                                         (lambda p=full: open(p, "rb").read(4)))))
            elif low.endswith(".rar"):
                problems.append(f"{name}: .rar isn't read")
            else:
                entries.append((".", Entry(name, os.path.getsize(full), (lambda p=full: open(p, "rb").read()),
                                         (lambda p=full: open(p, "rb").read(4)))))
        songs, others = group_songs(entries)
        for i, (kind, label, e) in enumerate(others):
            if kind == "maybe_con":
                magic = con_magic(e)
                others[i] = (magic or ("no_extension_unchecked" if e.head is None else "unknown_no_extension"), label, e)
        rows = [describe(s, problems) for s in songs]
        rows.sort(key=lambda r: (r["source"], r["folder"]))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(summary(rows, others, sources, problems))
        print(f"{len(rows)} songs; reports in {out_dir}", file=sys.stderr)
    finally:
        for zf in open_zips:
            zf.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
