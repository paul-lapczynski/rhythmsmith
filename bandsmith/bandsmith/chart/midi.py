"""notes.mid in the Rock Band / Clone Hero layout, written and read back.

Tracks: the tempo map (named after the song), PART GUITAR, PART BASS,
PART DRUMS, PART VOCALS, EVENTS (music_start, sections, music_end, end) and
BEAT (12 on the first beat of a bar, 13 on the others).

    5-fret     difficulty base + 0-4 (Expert 96, Hard 84, Medium 72, Easy 60),
               base + 5 forced HOPO, base + 6 forced strum, 116 star power
    drums      base + 0 kick, + 1-4 pads; 95 Expert+ kick; 110-112 tom markers
               for yellow/blue/green (cymbal otherwise: pro drums), 116 star power
    vocals     36-84 pitched notes, lyrics as text events on them ("#" spoken,
               "-" joins syllables, "+" a slide), 105 phrases, 116 star power
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import mido

from .model import (DIFFICULTIES, DRUMS, FRETTED, RESOLUTION, Note, Part, Phrase, Section, Song,
                    Tempo, TimeSig, VocalNote, VocalPart)

TRACK_NAMES = {"guitar": "PART GUITAR", "bass": "PART BASS", "drums": "PART DRUMS"}
BASE = {"expert": 96, "hard": 84, "medium": 72, "easy": 60}
STAR_POWER, PHRASE, DOUBLE_KICK = 116, 105, 95
ACTIVATION = range(120, 125)          # drum fills: all five, over the fill
TOMS = {2: 110, 3: 111, 4: 112}
# a .mid note this long or shorter is a plain note, not a sustain: Clone Hero's
# default for .mid (a third of a beat and a tick; scan-chart does the same). Rock
# Band charts write their plain notes a 16th long, so those must stay plain.
SHORT = RESOLUTION // 32
SUSTAIN_CUTOFF = RESOLUTION // 3 + 1


def _events_to_track(name: str, events: list[tuple]) -> mido.MidiTrack:
    """events: (tick, order, message) with absolute ticks; note-offs sort first."""
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=midi_text(name), time=0))
    last = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        tr.append(msg.copy(time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _note(events: list, tick: int, length: int, pitch: int, velocity: int = 100) -> None:
    events.append((tick, 1, mido.Message("note_on", note=pitch, velocity=velocity)))
    events.append((tick + max(length, 1), 0, mido.Message("note_off", note=pitch, velocity=0)))


_PLAIN = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "…": "..."})


def midi_text(text: str) -> str:
    """Text a .mid file can hold: MIDI text events are Latin-1, so curly quotes
    become straight ones, accents are kept where Latin-1 has them, and anything
    else is spelled without its accent or dropped."""
    import unicodedata
    text = text.translate(_PLAIN)
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        out = []
        for ch in text:
            try:
                ch.encode("latin-1")
                out.append(ch)
            except UnicodeEncodeError:
                out.append(unicodedata.normalize("NFKD", ch).encode("latin-1", "ignore").decode("latin-1"))
        return "".join(out)


def _text(events: list, tick: int, text: str, kind: str = "text") -> None:
    events.append((tick, 2, mido.MetaMessage(kind, text=midi_text(text))))


def _part_events(name: str, part: Part) -> list:
    ev = []
    for diff, notes in part.notes.items():
        base = BASE[diff]
        for n in notes:
            length = n.length if n.length > SUSTAIN_CUTOFF else SHORT
            for lane in n.lanes:
                pitch = DOUBLE_KICK if (name == DRUMS and lane == 0 and n.double_kick) else base + lane
                _note(ev, n.tick, length, pitch)
            if name == DRUMS:
                for lane in n.lanes:
                    if lane in TOMS and lane not in n.cymbals:
                        _note(ev, n.tick, SHORT, TOMS[lane])
            elif n.forced:
                _note(ev, n.tick, SHORT, base + (5 if n.forced == "hopo" else 6))
    for sp in part.star_power:
        _note(ev, sp.tick, sp.length, STAR_POWER)
    for a in part.activations:
        for p in ACTIVATION:
            _note(ev, a.tick, a.length, p)
    return _dedupe(ev)


def _dedupe(ev: list) -> list:
    """Two overlapping notes on one pitch (a tom marker under a chord, say)
    would cut each other short: keep the longest of each (tick, pitch)."""
    out = [e for e in ev if e[2].type not in ("note_on", "note_off")]
    by_pitch: dict[int, list[tuple[int, int]]] = {}
    pending: dict[int, list[int]] = {}
    for tick, order, msg in sorted(ev, key=lambda e: (e[0], -e[1])):
        if msg.type == "note_on":
            pending.setdefault(msg.note, []).append(tick)
        elif msg.type == "note_off":
            starts = pending.get(msg.note)
            if starts:
                by_pitch.setdefault(msg.note, []).append((starts.pop(0), tick))
    for pitch, spans in by_pitch.items():
        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s == merged[-1][0]:
                merged[-1][1] = max(merged[-1][1], e)
            elif merged and s < merged[-1][1]:
                merged[-1][1] = s                    # the earlier note ends where the next starts
                merged.append([s, e])
            else:
                merged.append([s, e])
        for s, e in merged:
            if e > s:
                _note(out, s, e - s, pitch)
    return out


def _vocal_events(v: VocalPart) -> list:
    ev = []
    for n in v.notes:
        pitch = n.pitch if n.pitch is not None else 60
        _note(ev, n.tick, max(n.length, SHORT), min(84, max(36, pitch)))
        lyric = n.lyric if n.pitch is not None or n.lyric.endswith("#") else n.lyric + "#"
        _text(ev, n.tick, lyric, "lyrics")
    for p in v.phrases:
        _note(ev, p.tick, p.length, PHRASE)
    for sp in v.star_power:
        _note(ev, sp.tick, sp.length, STAR_POWER)
    return _dedupe(ev)


def _beat_ticks(song: Song, end: int) -> list[tuple[int, bool]]:
    """(tick, is the first beat of a bar) for every beat up to `end`."""
    out = []
    sigs = sorted(song.time_sigs, key=lambda s: s.tick) or [TimeSig(0)]
    for i, ts in enumerate(sigs):
        stop = sigs[i + 1].tick if i + 1 < len(sigs) else end + 1
        step = song.resolution * 4 // ts.denominator
        t, k = ts.tick, 0
        while t < stop:
            out.append((t, k % ts.numerator == 0))
            t += step
            k += 1
    return out


def section_event(name: str) -> str:
    return f"[section {name}]"


def to_midi(song: Song) -> mido.MidiFile:
    mf = mido.MidiFile(type=1, ticks_per_beat=song.resolution)
    last = song.last_tick()
    end = song.end if song.end is not None else last + song.resolution * 2
    # the tempo map
    ev = []
    for t in sorted(song.tempos, key=lambda t: t.tick):
        ev.append((t.tick, 0, mido.MetaMessage("set_tempo", tempo=round(60_000_000 / t.bpm))))
    for ts in sorted(song.time_sigs, key=lambda s: s.tick):
        ev.append((ts.tick, 1, mido.MetaMessage("time_signature", numerator=ts.numerator,
                                                denominator=ts.denominator)))
    mf.tracks.append(_events_to_track(song.name or "notes", ev))
    for name in ("drums", "bass", "guitar"):
        if name in song.parts and any(song.parts[name].notes.values()):
            mf.tracks.append(_events_to_track(TRACK_NAMES[name], _part_events(name, song.parts[name])))
    if song.vocals and song.vocals.notes:
        mf.tracks.append(_events_to_track("PART VOCALS", _vocal_events(song.vocals)))
    # EVENTS
    ev = []
    first = song.first_tick() or 0
    start = song.music_start if song.music_start is not None else max(0, first - song.resolution * 4)
    _text(ev, start, "[music_start]")
    for s in song.sections:
        _text(ev, s.tick, section_event(s.name))
    _text(ev, max(last, end - song.resolution), "[music_end]")
    _text(ev, end, "[end]")
    mf.tracks.append(_events_to_track("EVENTS", ev))
    # BEAT
    ev = []
    for tick, down in _beat_ticks(song, end):
        _note(ev, tick, song.resolution // 4, 12 if down else 13)
    mf.tracks.append(_events_to_track("BEAT", ev))
    return mf


def write_mid(song: Song, path: Path) -> None:
    to_midi(song).save(str(path))


# ------------------------------------------------------------------ reading

PART_FOR = {v: k for k, v in TRACK_NAMES.items()}
SECTION_RE = re.compile(r"^\[(?:section[ _](.+)|prc_(.+))\]$")


def _spans(track) -> list[tuple[int, int, int, object]]:
    """(start tick, length, pitch, lyric/text at the start) for every note in a track."""
    t, opened, out = 0, {}, []
    for msg in track:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            opened.setdefault(msg.note, []).append(t)
        elif msg.type in ("note_off", "note_on"):
            starts = opened.get(msg.note)
            if starts:
                s = starts.pop(0)
                out.append((s, t - s, msg.note))
    return sorted(out)


def _texts(track) -> list[tuple[int, str, str]]:
    t, out = 0, []
    for msg in track:
        t += msg.time
        if msg.is_meta and msg.type in ("text", "lyrics"):
            out.append((t, msg.type, msg.text))
    return out


def _read_part(name: str, track) -> Part:
    spans = _spans(track)
    part = Part()
    by_diff: dict[str, dict[int, dict]] = {d: {} for d in DIFFICULTIES}
    markers: dict[int, set[int]] = {}
    for s, ln, p in spans:
        if p == STAR_POWER:
            part.star_power.append(Phrase(s, ln))
            continue
        if name == DRUMS and p in ACTIVATION:
            if p == ACTIVATION[0]:
                part.activations.append(Phrase(s, ln))
            continue
        if name == DRUMS and p in TOMS.values():
            markers.setdefault(s, set()).add(p)
            continue
        if name == DRUMS and p == DOUBLE_KICK:
            n = by_diff["expert"].setdefault(s, {"lanes": set(), "len": 0, "forced": None, "dk": False})
            n["lanes"].add(0)
            n["dk"] = True
            continue
        for diff, base in BASE.items():
            off = p - base
            if 0 <= off <= 6:
                n = by_diff[diff].setdefault(s, {"lanes": set(), "len": 0, "forced": None, "dk": False})
                if off <= 4:
                    n["lanes"].add(off)
                    n["len"] = max(n["len"], ln if ln > SUSTAIN_CUTOFF else 0)
                elif name in FRETTED:
                    n["forced"] = "hopo" if off == 5 else "strum"
                break
    for diff, notes in by_diff.items():
        out = []
        for tick in sorted(notes):
            n = notes[tick]
            if not n["lanes"]:
                continue
            cymbals = ()
            if name == DRUMS:
                toms = {lane for lane, m in TOMS.items() if m in markers.get(tick, set())}
                cymbals = tuple(sorted(lane for lane in n["lanes"] if lane in TOMS and lane not in toms))
            out.append(Note(tick, tuple(sorted(n["lanes"])), n["len"] if name != DRUMS else 0,
                            n["forced"], cymbals, n["dk"] and n["lanes"] == {0}))
        if out:
            part.notes[diff] = out
    return part


def _read_vocals(track) -> VocalPart:
    v = VocalPart()
    lyrics = {}
    for t, _, text in _texts(track):
        if not text.startswith("["):
            lyrics.setdefault(t, text)
    for s, ln, p in _spans(track):
        if p == PHRASE or p == 106:
            v.phrases.append(Phrase(s, ln))
        elif p == STAR_POWER:
            v.star_power.append(Phrase(s, ln))
        elif 36 <= p <= 84:
            lyric = lyrics.get(s, "")
            pitched = not lyric.endswith(("#", "^"))
            v.notes.append(VocalNote(s, ln, p if pitched else None, lyric.rstrip("#^") if not pitched else lyric))
    return v


def from_midi(mf: mido.MidiFile) -> Song:
    song = Song(resolution=mf.ticks_per_beat, tempos=[], time_sigs=[])
    for i, track in enumerate(mf.tracks):
        name = track.name
        if i == 0:
            song.name = name if name != "notes" else ""
            t = 0
            for msg in track:
                t += msg.time
                if msg.type == "set_tempo":
                    song.tempos.append(Tempo(t, round(mido.tempo2bpm(msg.tempo), 3)))
                elif msg.type == "time_signature":
                    song.time_sigs.append(TimeSig(t, msg.numerator, msg.denominator))
        elif name in PART_FOR:
            song.parts[PART_FOR[name]] = _read_part(PART_FOR[name], track)
        elif name == "PART VOCALS":
            song.vocals = _read_vocals(track)
        elif name == "EVENTS":
            for t, _, text in _texts(track):
                m = SECTION_RE.match(text.strip())
                if m:
                    song.sections.append(Section(t, (m.group(1) or m.group(2)).strip()))
                elif text.strip() == "[music_start]":
                    song.music_start = t
                elif text.strip() == "[end]":
                    song.end = t
    if not song.tempos:
        song.tempos = [Tempo(0, 120.0)]
    if not song.time_sigs:
        song.time_sigs = [TimeSig(0)]
    return song


def read_mid(path_or_bytes) -> Song:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return from_midi(mido.MidiFile(file=io.BytesIO(path_or_bytes)))
    return from_midi(mido.MidiFile(str(path_or_bytes)))
