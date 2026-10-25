"""From a song to a Clone Hero / YARG song folder, Expert only:

    notes.mid, song.ini, song.ogg (the rest of the band), guitar.ogg, bass.ogg,
    drums.ogg, vocals.ogg, and album.png when there's a picture

Stems come from the vocal separation add-on (htdemucs_6s, on the GPU when
there is one, see rhythmsmith_core.accel); without it, everything is charted
from the full mix and the folder gets one song.ogg. The opt-in add-ons improve
it when installed: Beat This! for the beats (ai_beats), basic-pitch for guitar
and bass notes, Whisper for the lyrics.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..chart import Part, Section, Song, VocalPart, tempo_map, write_ini, write_mid
from ..chart.model import TempoMap
from ..chart.tempo import beat_this_beats, detect_beats
from . import ai_model as ai, audio, charting, drums, music, pitched, vocals

LEAD_IN = 2.5                   # seconds before the first note (Clone Hero wants 2 or more)


class Cancelled(Exception):
    """make() was asked to stop (its `cancel` callable returned True)."""
ALL = ("guitar", "bass", "drums", "vocals")


@dataclass
class Result:
    folder: Path
    song: Song
    pad: float                                  # seconds of silence added in front
    timings: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)   # instrument -> Expert note count
    used: dict = field(default_factory=dict)    # what each part was made from
    music: dict = field(default_factory=dict)   # key, chord kinds, slides/bends/legato found


def tags(path: Path) -> tuple[str, str, str, str]:
    """(title, artist, album, year) from the file's tags, else its name."""
    title = artist = album = year = ""
    try:
        import mutagen
        f = mutagen.File(str(path), easy=True)
        t = (f.tags or {}) if f else {}
        title, artist = (t.get("title") or [""])[0], (t.get("artist") or [""])[0]
        album, year = (t.get("album") or [""])[0], (t.get("date") or [""])[0][:4]
    except Exception:
        pass
    if not title:
        stem = path.stem
        if " - " in stem:
            artist, title = (s.strip() for s in stem.split(" - ", 1))
        else:
            title = stem
    return title.strip(), artist.strip(), album.strip(), year.strip()


def sections(mix: np.ndarray, sr: int, beats: np.ndarray, tm: TempoMap, bars: list[int]) -> list[Section]:
    """Section starts where the music changes, on bar lines."""
    import librosa
    if len(beats) < 32:
        return [Section(bars[0] if bars else 0, "Intro")]
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=mix, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=mix, sr=sr, hop_length=hop, n_mfcc=13)
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    feats = librosa.util.sync(np.vstack([chroma, mfcc / 50]), frames)
    k = int(np.clip(len(beats) / 32, 3, 12))
    bounds = librosa.segment.agglomerative(feats, k)
    starts = []
    for b in bounds:
        b = min(b, len(beats) - 1)
        tick = tm.tick(beats[b])
        bar = min(bars, key=lambda t: abs(t - tick)) if bars else int(tick)
        if bar not in starts:
            starts.append(bar)
    starts.sort()
    names = ["Intro"] + [f"Part {i}" for i in range(2, len(starts) + 1)]
    return [Section(t, n) for t, n in zip(starts, names)]


def bar_ticks(song: Song, end: int) -> list[int]:
    out = []
    sigs = sorted(song.time_sigs, key=lambda s: s.tick)
    for i, ts in enumerate(sigs):
        stop = sigs[i + 1].tick if i + 1 < len(sigs) else end
        step = ts.numerator * song.resolution * 4 // ts.denominator
        out += list(range(ts.tick, stop, step))
    return out


def album_art(src: Path, title: str, artist: str, out: Path, online: bool, log=print) -> bool:
    """album.png (512x512) from art beside the song, in its tags, or (opted in) online."""
    from rhythmsmith_core import artwork
    artwork.set_app("BandSmith", __version__, "https://github.com/paul-lapczynski/rhythmsmith")
    data, where = artwork.find(src, title, artist, look_online=online, log=log)
    if not data:
        return False
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage
        img = QImage.fromData(data)
        if img.isNull():
            return False
        img = img.scaled(512, 512, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        img = img.copy((img.width() - 512) // 2, (img.height() - 512) // 2, 512, 512)
        ok = img.save(str(out / "album.png"), "PNG")
    except ImportError:
        import io
        from PIL import Image, ImageOps
        ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), (512, 512)).save(out / "album.png")
        ok = True
    if ok:
        log(f"album art: from {artwork.describe(where)}")
    return ok


def folder_name(title: str, artist: str) -> str:
    name = f"{artist} - {title}" if artist else title
    return re.sub(r'[<>:"/\\|?*]', "", name).strip(" .") or "song"


def make(src: Path, out: Path, instruments=ALL, stems_dir: Path | None = None, cache: Path | None = None,
         art_online: bool = False, ai_beats: bool = False, ai_notes: bool = True, whisper: bool = True,
         title: str | None = None, artist: str | None = None, log=print, cancel=None,
         ai_model: bool = False) -> Result:
    """Chart `src` into the song folder `out` (made if missing). `cancel`: a callable,
    checked between steps; when it returns True, make() stops with Cancelled. `ai_model`:
    place guitar, bass and drums with the AI model where its file is installed
    (analyze.ai_model) and there are stems; the rules otherwise."""

    def check():
        if cancel is not None and cancel():
            raise Cancelled()
    t0 = time.time()
    timings = {}
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache) if cache else out.parent / ".bandsmith-cache" / out.name     # not inside the song folder
    t_title, t_artist, album, year = tags(src)
    title, artist = title or t_title, artist if artist is not None else t_artist
    used = {}

    stem_files = audio.stems(src, cache, guitar="guitar" in instruments, given=stems_dir, log=log)
    timings["stems"] = time.time() - t0
    check()
    log("progress: loading audio")
    t = time.time()
    mix = audio.load(src)
    sr = audio.SR
    st = {n: audio.load(p) for n, p in stem_files.items()}
    # guitar: its own stem when it has something in it, else "other" (what htdemucs puts guitars in)
    guitar_src = None
    if "guitar" in st and ("other" not in st or audio.rms(st["guitar"]) > 0.25 * audio.rms(st["other"])):
        guitar_src = "guitar"
    elif "other" in st:
        guitar_src = "other"
    timings["load"] = time.time() - t
    check()
    log("progress: finding the beats")

    t = time.time()
    found = beat_this_beats(src, log) if ai_beats else None
    if found:
        beats, downs = found
        used["beats"] = "Beat This!"
    else:
        beats, downs = detect_beats(mix, sr, low=st.get("drums"))
        used["beats"] = "librosa"
    timings["beats"] = time.time() - t
    check()

    t = time.time()
    raw: dict = {}
    if "drums" in instruments:
        log("progress: drums")
        raw["drums"] = drums.hits(st.get("drums", mix), sr, sensitivity=1.0 if "drums" in st else 0.7)
        used["drums"] = "drums stem" if "drums" in st else "full mix"
    check()
    if "guitar" in instruments:
        log("progress: guitar")
        y = st.get(guitar_src) if guitar_src else mix
        raw["guitar"] = pitched.events(stem_files.get(guitar_src), y, "guitar", ai_notes, sr, log)
        used["guitar"] = f"{guitar_src or 'full mix'} ({'basic-pitch' if ai_notes and pitched.basic_pitch_available() else 'pYIN'})"
    check()
    if "bass" in instruments:
        log("progress: bass")
        y = st.get("bass")
        if y is None:
            import librosa
            y = librosa.effects.preemphasis(mix, coef=-0.95)       # tilt toward the low end
        raw["bass"] = pitched.events(stem_files.get("bass"), y, "bass", ai_notes, sr, log)
        used["bass"] = "bass stem" if "bass" in st else "full mix"
    has_lyrics = False
    check()
    if "vocals" in instruments and "vocals" in st:
        log("progress: vocals")
        import librosa
        y16 = librosa.resample(st["vocals"], orig_sr=sr, target_sr=vocals.SR_V)
        raw["vocals"], has_lyrics = vocals.notes(stem_files["vocals"], y16, whisper, log)
        used["vocals"] = "vocals stem" + (" + Whisper lyrics" if has_lyrics else "")
    timings["transcribe"] = time.time() - t
    check()
    log("progress: charting")

    # silence in front, so the first note is LEAD_IN seconds in
    firsts = [h.time for h in raw.get("drums", [])[:1]] + [e.start for k in ("guitar", "bass") for e in raw.get(k, [])[:1]]
    firsts += [v.start for v in raw.get("vocals", [])[:1]]
    pad = round(max(0.0, LEAD_IN - min(firsts)), 3) if firsts else 0.0
    beats = np.asarray(beats) + pad
    downs = np.asarray(downs) + pad
    for h in raw.get("drums", []):
        h.time += pad
    for k in ("guitar", "bass"):
        for e in raw.get(k, []):
            e.start += pad
            e.end += pad
    for v in raw.get("vocals", []):
        v.start += pad
        v.end += pad

    # beat trackers place beats to a frame (librosa ~23 ms, Beat This! 20 ms): don't chase that jitter
    tempos, sigs, _ = tempo_map(beats, downs, tolerance=0.025)
    song = Song(name=title, artist=artist, album=album, year=year, charter=f"BandSmith {__version__}",
                tempos=tempos, time_sigs=sigs)
    tm = song.tempo_map()
    duration = len(mix) / sr + pad

    def model_notes(name):
        """The AI model's notes for `name`, or None to use the rules."""
        src_stem = guitar_src if name == "guitar" else name
        if not (ai_model and src_stem in st and ai.available(name)):
            return None
        log(f"progress: {name} (AI model)")
        used[name] = f"{used.get(name, '')} + AI model".lstrip(" +")
        return ai.predict(name, st[src_stem], mix, sr, tm, pad, int(tm.tick(duration)))

    def fretted(name):
        """Guitar or bass from the notes that were played (music.py); with the AI
        model, only those it also hears start."""
        src_stem = guitar_src if name == "guitar" else name
        y = st.get(src_stem)
        ps = music.merge_held(music.played(raw[name], y if y is not None else mix, sr, offset=pad))
        if y is not None:
            log(f"progress: {name} slides and bends")
            ps, g = music.gestures(ps, music.contour(y, sr, name, offset=pad))
            gestures[name] = g
        if ai_model and y is not None and ai.available(name):
            out, meta = ai.run(name, y, mix, sr, tm, pad, int(tm.tick(duration)))
            ps = music.confirm(ps, tm, out[:, 0], ai.onset_threshold(meta))
            used[name] = f"{used.get(name, '')} + AI model".lstrip(" +")
        ns, info = music.chart(ps, tm, bass=name == "bass")
        chords[name] = info
        return ns

    gestures, chords = {}, {}
    for name in ("guitar", "bass"):
        if name in raw:
            check()
            ns = fretted(name)
            song.parts[name] = Part(charting.all_difficulties(ns), charting.star_power(ns))
    if "drums" in raw:
        ns = model_notes("drums")
        if ns is None:
            ns = charting.drum_notes(raw["drums"], tm)
        sp = charting.star_power(ns)
        song.parts["drums"] = Part(charting.all_difficulties(ns, drums=True), sp, charting.drum_fills(ns, sp))
    if raw.get("vocals"):
        vn, phrases = charting.vocal_notes(raw["vocals"], tm)
        if vn:
            vp = VocalPart(vn, phrases)
            vp.star_power = [p for p in phrases[2::4]]
            song.vocals = vp
    end_tick = int(tm.tick(duration - 0.5))
    song.end = max(song.last_tick() + 1, end_tick)
    bars = bar_ticks(song, song.end)
    first_note = song.first_tick() or 0
    song.sections = [s for s in sections(np.concatenate([np.zeros(int(pad * sr), np.float32), mix]), sr,
                                         beats, tm, bars) if s.tick < song.end]
    if song.sections and song.sections[0].tick > first_note:
        song.sections[0].tick = max(0, max(b for b in bars if b <= first_note) if bars else 0)
    song.music_start = max(0, first_note - 4 * 480)

    check()
    log("progress: writing the song folder")
    t = time.time()
    if stem_files:
        names = {"guitar": guitar_src, "bass": "bass", "drums": "drums", "vocals": "vocals"}
        written = set()
        for out_name, stem in names.items():
            if out_name in instruments and stem in stem_files:
                audio.write_padded(stem_files[stem], out / f"{out_name}.ogg", pad)
                written.add(stem)
        rest = [stem_files[n] for n in stem_files if n not in written]
        if rest:
            ys = [audio.load(p, audio.OUT_SR, mono=False) for p in rest]
            n = max(len(y) for y in ys)
            y = np.zeros((n, 2), np.float32)
            for part in ys:
                y[: len(part)] += part
            y = np.concatenate([np.zeros((int(round(pad * audio.OUT_SR)), 2), np.float32), y])
            audio.write_ogg(out / "song.ogg", y)
    else:
        audio.write_padded(src, out / "song.ogg", pad)
    write_mid(song, out / "notes.mid")
    tiers = {n: charting.tier(p.notes.get("expert", []), tm) for n, p in song.parts.items()}
    tiers["vocals"] = 2
    write_ini(song, out / "song.ini", int(duration * 1000), difficulties=tiers)
    album_art(src, title, artist, out, art_online, log)
    timings["write"] = time.time() - t
    timings["total"] = time.time() - t0
    counts = {n: len(p.notes.get("expert", [])) for n, p in song.parts.items()}
    if song.vocals:
        counts["vocals"] = len(song.vocals.notes)
    info = {"gestures": gestures, "chords": chords}
    try:
        info["key"] = music.song_key(mix, sr)
    except Exception:
        pass
    return Result(out, song, pad, timings, counts, used, info)
