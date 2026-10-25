"""Guitar and bass charting that follows the music: only notes that were
actually played (transcribed), held notes as one sustained note, chords by
their intervals, frets from pitch, and repeated riffs charted the same.

    1. The transcription is the source of truth: every chart note is a
       transcribed note (pitch, start, end). Nothing is added for the rhythm.
    2. A pitch that keeps ringing with no new attack is one note, however many
       times the transcriber split it, sustained as long as it rings.
    3. With the AI model, a transcribed note is kept only where the model also
       hears a note start (its onset probability near that slot).
    4. A chord only where the stem really sounds two or more different pitches
       at once (starting within a strum's spread). Its size comes from the
       voicing: root and fifth (or fourth, or octave), two frets; a third in it
       or three or more pitch classes, three frets; four or more different
       pitches, four frets (capped by max_chord). The lowest pitch is the
       leftmost fret. Bass is charted one note at a time.
    5. Frets come from pitch within a phrase (notes with less than a beat
       between them): the same pitch is always the same fret, higher pitches
       are further right, in a 3-5 fret window placed by the phrase's pitch in
       the song's range. The window only moves between phrases.
    6. A phrase that repeats an earlier one (the same pitches on the same
       rhythm) gets the same frets.
    7. Gestures, from the stem's pitch contour and attacks: a bend or vibrato
       (the pitch moves and comes back, no new attack) stays one sustained
       note; a slide (a glide that settles on a new note, no new attack) is the
       held note, then the arrival as a forced HOPO on a fret in the glide's
       direction; any other new pitch without a new attack (a hammer-on or
       pull-off) is a HOPO too; a new attack is a normal note.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chart.model import Note
from .pitched import Event

R = 480
SIXTEENTH = R // 4
MIN_SUSTAIN = R // 2             # shorter rings are plain notes
SUSTAIN_GAP = SIXTEENTH          # a sustain ends this far before the next note
CHORD_WINDOW = 0.045             # seconds: pitches starting this close are one chord
HELD_GAP = 0.06                  # seconds: a "new" note this soon after the last one ended may be the same note
NEW_ATTACK = 0.35                # onset strength (0-1 of the song's strongest) that counts as a new attack
KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


@dataclass
class Played:
    """A transcribed note or chord, in seconds."""
    start: float
    end: float
    pitches: list[int]
    strength: float = 1.0
    attack: float = 1.0           # onset strength of the stem at its start, 0-1
    hopo: bool = False            # reached without a new attack (slide, hammer-on, pull-off)
    gesture: str = ""             # "slide", "legato" or "" (and "bend" on notes that absorbed one)


# ------------------------------------------------------------------ the notes that were played

def attacks(y: np.ndarray, sr: int, times: list[float]) -> list[float]:
    """The stem's onset strength (0-1 of its strongest) at each time."""
    import librosa
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    top = float(np.percentile(env, 99)) or 1.0
    out = []
    for t in times:
        f = int(round(t * sr / hop))
        out.append(float(env[max(0, f - 2):f + 3].max() / top) if len(env) else 0.0)
    return out


def played(events: list[Event], y: np.ndarray | None = None, sr: int = 22050, offset: float = 0.0) -> list[Played]:
    """Transcribed events as notes, with how hard each was attacked. `offset`:
    seconds the event times are ahead of the audio `y` (the chart's lead-in)."""
    ps = [Played(e.start, e.end, sorted(set(e.pitches)), e.strength) for e in events if e.pitches]
    if y is not None and ps:
        for p, a in zip(ps, attacks(y, sr, [p.start - offset for p in ps])):
            p.attack = a
    return ps


def merge_held(ps: list[Played]) -> list[Played]:
    """A note that repeats pitches that are still ringing, without a new attack,
    is the same note: extend it instead."""
    out: list[Played] = []
    for p in sorted(ps, key=lambda p: p.start):
        if out:
            q = out[-1]
            still = p.start <= q.end + HELD_GAP
            same = set(p.pitches) <= set(q.pitches)
            if still and same and p.attack < NEW_ATTACK:
                q.end = max(q.end, p.end)
                continue
        out.append(p)
    return out


def confirm(ps: list[Played], tm, onset_prob, threshold: float) -> list[Played]:
    """Keep the notes the AI model also hears start: its onset probability (one
    per 1/12-beat slot) within a 16th of the note."""
    keep = []
    for p in ps:
        s = int(round(tm.tick(p.start) / 40))
        if onset_prob[max(0, s - 3):s + 4].max(initial=0.0) >= threshold:
            keep.append(p)
    return keep


# ------------------------------------------------------------------ gestures

@dataclass
class Contour:
    times: np.ndarray
    midi: np.ndarray              # fractional MIDI pitch, nan where unvoiced

    def span(self, a: float, b: float) -> np.ndarray:
        m = (self.times >= a) & (self.times < b)
        return self.midi[m]


def contour(y: np.ndarray, sr: int, kind: str = "guitar", offset: float = 0.0) -> Contour:
    """The stem's main pitch over time (pYIN, about every 23 ms), on the chart's
    clock (`offset` seconds of lead-in added)."""
    import librosa
    lo, hi = (75.0, 1000.0) if kind == "guitar" else (30.0, 400.0)
    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=lo, fmax=hi, sr=sr, hop_length=hop,
                                 frame_length=2048 if kind == "guitar" else 4096)
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    return Contour(librosa.times_like(f0, sr=sr, hop_length=hop) + offset, midi)


def gestures(ps: list[Played], c: Contour) -> tuple[list[Played], dict]:
    """Bends folded into the note they bend; slides and legato notes marked HOPO."""
    out: list[Played] = []
    counts = {"bend": 0, "slide": 0, "legato": 0}
    for p in sorted(ps, key=lambda p: p.start):
        q = out[-1] if out else None
        if q is None or p.attack >= NEW_ATTACK or p.start > q.end + HELD_GAP or len(p.pitches) > 1:
            out.append(p)
            continue
        step = p.pitches[0] - q.pitches[0]
        if step == 0:
            out.append(p)
            continue
        tail = c.span(p.start + 0.6 * (p.end - p.start), p.end + 0.05)
        tail = tail[~np.isnan(tail)]
        if abs(step) <= 2 and len(tail) and abs(np.median(tail) - q.pitches[0]) < 0.6:
            # the pitch went up (or wobbled) and came back: one note, held through
            q.end = max(q.end, p.end)
            q.gesture = "bend"
            counts["bend"] += 1
            continue
        between = c.span(p.start - 0.12, p.start + 0.04)
        voiced = between[~np.isnan(between)]
        gliding = len(voiced) >= 0.8 * max(1, len(between)) and len(voiced) >= 3 and             np.all(np.sign(np.diff(voiced)) * np.sign(step) >= -0.05)
        p.hopo = True
        p.gesture = "slide" if gliding else "legato"
        counts[p.gesture] += 1
        q.end = min(q.end, p.start) if p.gesture == "slide" else q.end
        out.append(p)
    return out, counts


# ------------------------------------------------------------------ harmony

def chord_kind(pitches: list[int]) -> str:
    """"single", "power" (root and fifth/fourth/octave), "full" (a third in it, or
    three or more pitch classes) or "dense" (four or more different pitches)."""
    pitches = sorted(set(pitches))
    if len(pitches) < 2:
        return "single"
    root = pitches[0]
    classes = {(p - root) % 12 for p in pitches}
    if classes == {0}:                                         # octaves of one note: one fret
        return "single"
    if classes <= {0, 7, 5}:
        return "power"
    if len(pitches) >= 4 and len(classes) >= 3:
        return "dense"
    return "full"


CHORD_SIZE = {"single": 1, "power": 2, "full": 3, "dense": 4}


def song_key(y: np.ndarray, sr: int) -> str:
    """The likeliest key (Krumhansl-Schmuckler on the chroma), e.g. "E minor"."""
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = max(((np.corrcoef(np.roll(prof, k), chroma)[0, 1], k, name)
                for prof, name in ((major, "major"), (minor, "minor")) for k in range(12)))
    return f"{KEY_NAMES[best[1]]} {best[2]}"


# ------------------------------------------------------------------ frets

def quantize(tick: float) -> int:
    from .charting import quantize as q
    return q(tick)


def phrases(ticks: list[int]) -> list[list[int]]:
    """Indexes of notes grouped where a beat or more passes between them."""
    out, cur = [], []
    for i, t in enumerate(ticks):
        if cur and t - ticks[cur[-1]] >= R:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def fret_map(roots: list[int], lo: float, hi: float) -> dict[int, int]:
    """Pitch -> fret for one phrase: same pitch same fret, higher pitch further
    right, in a window as wide as the phrase needs (3-5), placed by the phrase's
    pitch within the song's range."""
    uniq = sorted(set(roots))
    width = int(np.clip(len(uniq), 3, 5))
    center = (np.median(roots) - lo) / max(1.0, hi - lo)
    start = int(np.clip(round(center * (5 - width)), 0, 5 - width))
    if len(uniq) == 1:
        return {uniq[0]: start + width // 2}
    if len(uniq) <= width:
        # keep the steps between pitches when they fit: spread by pitch, not rank
        span = uniq[-1] - uniq[0]
        frets = [start + int(round((p - uniq[0]) / span * (width - 1))) for p in uniq]
        # neighbours that collapsed onto one fret: push apart in order (still left to right)
        for i in range(1, len(frets)):
            frets[i] = max(frets[i], frets[i - 1] + 1)
        if frets[-1] > 4:
            frets = [f - (frets[-1] - 4) for f in frets]
        return dict(zip(uniq, frets))
    # more pitches than frets: by pitch, so steps stay in order (equal at the edges)
    span = uniq[-1] - uniq[0]
    return {p: start + int(round((p - uniq[0]) / span * (width - 1))) for p in uniq}


def chart(ps: list[Played], tm, bass: bool = False, max_chord: int = 4) -> tuple[list[Note], dict]:
    """Expert chart notes from the played notes; and what was done (counts).
    max_chord: the largest chord, in frets (1-4)."""
    items = []
    for p in sorted(ps, key=lambda p: p.start):
        t = quantize(tm.tick(p.start))
        if items and t - items[-1][0] < SIXTEENTH:            # two starts in one 16th: keep the stronger
            if p.strength > items[-1][1].strength:
                items[-1] = (t, p)
            continue
        items.append((t, p))
    if not items:
        return [], {}
    roots = [p.pitches[0] for _, p in items]
    lo, hi = np.percentile(roots, 5), np.percentile(roots, 95)
    ticks = [t for t, _ in items]
    frets = [0] * len(items)
    seen: dict[tuple, list[int]] = {}
    repeats = 0
    for ph in phrases(ticks):
        key = tuple((ticks[i] - ticks[ph[0]], roots[i]) for i in ph)
        if key in seen:                                        # a riff we've charted: the same frets
            for i, f in zip(ph, seen[key]):
                frets[i] = f
            repeats += 1
            continue
        m = fret_map([roots[i] for i in ph], lo, hi)
        for i in ph:
            frets[i] = m[roots[i]]
        seen[key] = [frets[i] for i in ph]
    notes: list[Note] = []
    kinds = {"single": 0, "power": 0, "full": 0, "dense": 0}
    for k, (t, p) in enumerate(items):
        kind = "single" if bass else chord_kind(p.pitches)
        kinds[kind] += 1
        size = min(CHORD_SIZE[kind], max_chord)
        base = min(frets[k], 5 - size)
        lanes = tuple(range(base, base + size))
        length = int(round(tm.tick(p.end) - t))
        if k + 1 < len(items):
            length = min(length, items[k + 1][0] - t - SUSTAIN_GAP)
        length = (length // SIXTEENTH) * SIXTEENTH
        forced = None
        if p.hopo and notes:
            forced = "hopo"
            prev = notes[-1]
            step = p.pitches[0] - items[k - 1][1].pitches[0]
            if lanes == prev.lanes and step:
                # a slide or hammer-on lands on another fret, in its direction
                shift = 1 if step > 0 else -1
                if 0 <= lanes[0] + shift and lanes[-1] + shift <= 4:
                    lanes = tuple(x + shift for x in lanes)
        notes.append(Note(t, lanes, length if length >= MIN_SUSTAIN else 0, forced=forced))
    return notes, {"kinds": kinds, "repeated_phrases": repeats,
                   "gestures": [(n.tick, p.gesture, int(tm.tick(p.end))) for n, (_, p) in zip(notes, items)]}


# ------------------------------------------------------------------ checks

def held_changes(notes: list[Note], ps: list[Played], tm) -> int:
    """Chart notes that start while the previous note's pitch is still ringing
    with no new attack but on different frets (should be 0)."""
    starts = {quantize(tm.tick(p.start)): p for p in ps}
    bad = 0
    for a, b in zip(notes, notes[1:]):
        pa, pb = starts.get(a.tick), starts.get(b.tick)
        if pa and pb and pb.start <= pa.end + HELD_GAP and set(pb.pitches) <= set(pa.pitches) \
                and pb.attack < NEW_ATTACK and a.lanes != b.lanes:
            bad += 1
    return bad


def same_pitch_same_fret(notes: list[Note], ps: list[Played], tm) -> float:
    """Within each phrase, the share of repeated pitches that land on the fret
    that pitch first had."""
    by_tick = {quantize(tm.tick(p.start)): p.pitches[0] for p in ps}
    ticks = [n.tick for n in notes]
    same = total = 0
    for ph in phrases(ticks):
        first: dict[int, tuple] = {}
        for i in ph:
            pitch = by_tick.get(ticks[i])
            if pitch is None:
                continue
            if pitch in first:
                total += 1
                same += first[pitch] == notes[i].lanes
            else:
                first[pitch] = notes[i].lanes
    return same / total if total else 1.0
