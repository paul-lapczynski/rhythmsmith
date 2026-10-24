"""Expert charting rules: from note events and drum hits in seconds to chart
notes on the tempo grid.

  - every note goes to the nearest 16th (120 ticks) or, when clearly closer,
    12th-note triplet (160 ticks); notes closer than a 16th merge
  - frets follow the pitch contour: each phrase (notes with less than a beat
    between them) uses a window of 3 or 4 frets, placed by the phrase's pitch
    within the song's range, so higher notes are higher frets
  - chords have at most 3 notes, in power-chord shapes (adjacent frets)
  - a sustain is kept when the note rings at least half a beat, and ends a 16th
    before the next note
  - drums: two hands and a foot at once, pro-drum cymbal flags
  - star power every 8 bars or so, drum fills before it, sections from the music
"""
from __future__ import annotations

import numpy as np

from ..chart.model import Note, Phrase, VocalNote
from .drums import Hit, limb_limit, to_chords
from .pitched import Event
from .vocals import VNote

R = 480
SIXTEENTH, TRIPLET = R // 4, R // 3
MIN_GAP = SIXTEENTH                 # notes closer than this merge
MIN_SUSTAIN = R // 2
SUSTAIN_GAP = SIXTEENTH             # a sustain ends this far before the next note
TRIPLET_MARGIN = 20                 # ticks a triplet slot must win by


def quantize(tick: float) -> int:
    q16 = round(tick / SIXTEENTH) * SIXTEENTH
    q12 = round(tick / TRIPLET) * TRIPLET
    return int(q12 if abs(tick - q12) + TRIPLET_MARGIN < abs(tick - q16) else q16)


def _merge_close(items: list[tuple[int, object, float]]) -> list[tuple[int, object, float]]:
    """(tick, payload, strength) sorted; of notes closer than MIN_GAP keep the stronger."""
    out: list[tuple[int, object, float]] = []
    for it in sorted(items, key=lambda x: x[0]):
        if out and it[0] - out[-1][0] < MIN_GAP:
            if it[2] > out[-1][2]:
                out[-1] = it
            continue
        out.append(it)
    return out


def fret_notes(evs: list[Event], tm, window: int | None = None) -> list[Note]:
    """Guitar or bass notes from pitched events."""
    items = _merge_close([(quantize(tm.tick(e.start)), e, e.strength) for e in evs if e.pitches])
    if not items:
        return []
    roots = np.array([e.pitches[0] for _, e, _ in items], float)
    lo, hi = np.percentile(roots, 5), np.percentile(roots, 95)
    span = max(1.0, hi - lo)
    # phrases: split where there is a beat or more of space
    phrases, cur = [], [0]
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] >= R:
            phrases.append(cur)
            cur = []
        cur.append(i)
    phrases.append(cur)
    frets = [0] * len(items)
    for ph in phrases:
        ps = roots[ph]
        w = window or (4 if ps.max() - ps.min() >= 7 else 3)
        center = (np.median(ps) - lo) / span
        start = int(np.clip(round(center * (5 - w)), 0, 5 - w))
        pmin, pmax = ps.min(), ps.max()
        for i in ph:
            rel = 0.5 if pmax == pmin else (roots[i] - pmin) / (pmax - pmin)
            frets[i] = start + int(round(rel * (w - 1)))
    notes: list[Note] = []
    for k, (tick, e, _) in enumerate(items):
        f = frets[k]
        lanes: tuple[int, ...] = (f,)
        if len(e.pitches) >= 2:
            interval = e.pitches[-1] - e.pitches[0]
            size = 3 if interval >= 12 and len(e.pitches) >= 3 else 2
            base = min(f, 5 - size)
            lanes = tuple(range(base, base + size))
        nxt = items[k + 1][0] if k + 1 < len(items) else None
        length = int(round(tm.tick(e.end) - tick))
        if nxt is not None:
            length = min(length, nxt - tick - SUSTAIN_GAP)
        length = (length // SIXTEENTH) * SIXTEENTH
        notes.append(Note(tick, lanes, length if length >= MIN_SUSTAIN else 0))
    return notes


def drum_notes(hs: list[Hit], tm) -> list[Note]:
    chords = [limb_limit(c) for c in to_chords(hs)]
    items = _merge_close([(quantize(tm.tick(c[0].time)), c, sum(h.strength for h in c)) for c in chords if c])
    out = []
    for tick, c, _ in items:
        lanes = tuple(sorted({h.lane for h in c}))
        cymbals = tuple(sorted({h.lane for h in c if h.cymbal and h.lane >= 2}))
        out.append(Note(tick, lanes, cymbals=cymbals))
    return out


def vocal_notes(vs: list[VNote], tm) -> tuple[list[VocalNote], list[Phrase]]:
    """Vocal notes on the 16th grid, and phrases (notes with less than a beat
    between them, at most 4 bars long)."""
    notes: list[VocalNote] = []
    for v in sorted(vs, key=lambda v: v.start):
        s = round(tm.tick(v.start) / SIXTEENTH) * SIXTEENTH
        e = max(s + SIXTEENTH, round(tm.tick(v.end) / SIXTEENTH) * SIXTEENTH)
        if notes and s < notes[-1].tick + notes[-1].length:
            if s <= notes[-1].tick:
                continue
            notes[-1].length = s - notes[-1].tick
        notes.append(VocalNote(int(s), int(e - s), v.pitch, v.lyric))
    # a "+" can't start a phrase: make it a plain syllable
    phrases: list[Phrase] = []
    start = None
    for i, n in enumerate(notes):
        if start is None:
            start = i
        last = i + 1 == len(notes)
        gap = None if last else notes[i + 1].tick - (n.tick + n.length)
        too_long = (n.tick + n.length - notes[start].tick) >= 16 * R
        if last or gap >= R or too_long:
            s, e = notes[start].tick, n.tick + n.length
            phrases.append(Phrase(max(0, s - SIXTEENTH // 2), e - s + SIXTEENTH))
            if notes[start].lyric == "+":
                notes[start].lyric = "ah"
            start = None
    # phrases mustn't overlap
    for a, b in zip(phrases, phrases[1:]):
        if a.tick + a.length > b.tick:
            a.length = b.tick - a.tick
    return notes, phrases


def star_power(notes: list[Note], every_bars: int = 8, length_bars: int = 2, first_bar: int = 4) -> list[Phrase]:
    """A star power phrase every `every_bars` bars, over the notes of `length_bars`."""
    if len(notes) < 16:
        return []
    out = []
    t = notes[0].tick + first_bar * 4 * R
    end = notes[-1].tick
    while t < end:
        span = [n for n in notes if t <= n.tick < t + length_bars * 4 * R]
        if len(span) >= 3:
            s = span[0].tick
            e = span[-1].tick + max(span[-1].length, 1)
            out.append(Phrase(s, e - s + 1))
        t += every_bars * 4 * R
    return out


def drum_fills(notes: list[Note], sp: list[Phrase]) -> list[Phrase]:
    """Fills (where drummers activate star power): the last beat before every
    other star power phrase, when there are notes in it."""
    out = []
    for p in sp[1::2]:
        s, e = p.tick - R, p.tick - 1
        span = [n for n in notes if s <= n.tick <= e]
        if span:
            out.append(Phrase(span[0].tick, span[-1].tick - span[0].tick + 1))
    return out


def tier(notes: list[Note], tm) -> int:
    """A 0-6 difficulty tier from notes per second."""
    if len(notes) < 2:
        return 0
    secs = tm.seconds(notes[-1].tick) - tm.seconds(notes[0].tick)
    nps = len(notes) / max(1.0, secs)
    return int(np.clip(round(nps / 1.4), 0, 6))
