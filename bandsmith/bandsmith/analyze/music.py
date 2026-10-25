"""Guitar and bass charting that follows the music: only notes that were
actually played (transcribed), held notes as one sustained note, chords by
their intervals, frets from pitch, and repeated riffs charted the same.

    1. A note only where one is played: where a note starts (the AI model's
       onsets when it's on, else the stem's own onsets), with the pitches the
       transcription (basic-pitch, or pYIN) hears there. Nothing is added for
       the rhythm, and a start with no pitch heard is dropped, unless it's a
       sure one (a palm-muted chug), which repeats the last pitch.
    2. A pitch that keeps ringing with no new attack is one note, sustained as
       long as it rings (or, with the AI model, as long as the model calls it
       held); a re-pick while it rings keeps the same frets.
    3. A sustain ends a 16th before the next note: a repeated pluck is never
       one long hold.
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
       note; a slide (a voiced glide that settles on a new note, no new attack)
       is the held note, then the arrival as a forced HOPO on a fret in the
       glide's direction; hammer-ons and pull-offs are left to Clone Hero's
       own HOPO rule (close notes on different frets); a new attack is a
       normal note.
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
ONSET_DELTA = {"guitar": 0.05, "bass": 0.05}   # how sure a stem onset must be, without the AI model


@dataclass
class Played:
    """A transcribed note or chord, in seconds."""
    start: float
    end: float
    pitches: list[int]
    strength: float = 1.0
    attack: float = 1.0           # onset strength of the stem at its start, 0-1
    hopo: bool = False            # reached by a slide: a forced HOPO
    gesture: str = ""             # "slide", "legato" or "" (and "bend" on notes that absorbed one)
    held: bool | None = None      # the AI model's call on whether it's held (None: by its length)


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


def onsets(y: np.ndarray, sr: int, offset: float = 0.0, delta: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Where the stem's notes start (its onsets), on the chart's clock, and how hard (0-1)."""
    import librosa
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    top = float(np.percentile(env, 99)) or 1.0
    frames = librosa.onset.onset_detect(onset_envelope=env / top, sr=sr, hop_length=hop, delta=delta,
                                        backtrack=False, units="frames")
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop) + offset, env[frames] / top


def from_onsets(times, strengths, events: list[Event], c: "Contour | None" = None,
                near: float = 0.07, held=None, sure: float = 0.6, held_cut: float = 0.35) -> list[Played]:
    """A note at each onset, with the pitches the transcription hears there: the
    ones starting within `near` seconds (a chord if several), else the ones still
    ringing (a re-struck note), else the contour's pitch. With no pitch heard, a
    sure onset (strength >= `sure`, e.g. a palm-muted chug) repeats the last
    note's pitches; otherwise no note. `held`: the AI model's sustained-or-not
    probability per onset, if it has one."""
    evs = sorted(events, key=lambda e: e.start)
    starts = np.array([e.start for e in evs]) if evs else np.zeros(0)
    out: list[Played] = []
    order = np.argsort(times)
    times, strengths = np.asarray(times)[order], np.asarray(strengths)[order]
    for k, (t, s) in enumerate(zip(times, strengths)):
        nxt = times[k + 1] if k + 1 < len(times) else t + 4.0
        i0, i1 = np.searchsorted(starts, t - near), np.searchsorted(starts, t + near)
        here = evs[i0:i1]
        if not here:
            here = [e for e in evs[max(0, i0 - 40):i0] if e.start < t < e.end - 0.03]
        if here:
            pitches = sorted({p for e in here for p in e.pitches})
            end = max(e.end for e in here)
        elif c is not None:
            seg = c.span(t, min(nxt, t + 0.15))
            seg = seg[~np.isnan(seg)]
            if len(seg) < 3:
                if out and s >= sure:                  # a sure note with no pitch: the same as the last
                    out.append(Played(float(t), float(min(nxt, t + 0.15)), list(out[-1].pitches), float(s), float(s)))
                continue
            pitches = [int(round(float(np.median(seg))))]
            end = min(nxt, t + 0.25)
        elif out and s >= sure:
            pitches = list(out[-1].pitches)
            end = min(nxt, t + 0.15)
        else:
            continue
        p = Played(float(t), float(max(t + 0.05, end)), pitches, float(s), float(s))
        if held is not None:
            p.held = bool(held[order[k]] >= held_cut)
        out.append(p)
    return out


def merge_held(ps: list[Played], trust: float = 0.6) -> list[Played]:
    """A note that repeats pitches that are still ringing, without a new attack,
    is the same note: extend it instead. A note start the AI model is sure of
    (strength >= `trust`) is a re-strike and stays, whatever the stem's attack."""
    out: list[Played] = []
    for p in sorted(ps, key=lambda p: p.start):
        if out:
            q = out[-1]
            still = p.start <= q.end + HELD_GAP
            same = set(p.pitches) <= set(q.pitches)
            if still and same and p.attack < NEW_ATTACK and p.strength < trust:
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
        if (abs(step) <= 2 and p.attack < 0.2 and p.strength < 0.6 and len(tail)
                and abs(np.median(tail) - q.pitches[0]) < 0.6):
            # the pitch went up (or wobbled) and came back: one note, held through
            q.end = max(q.end, p.end)
            q.gesture = "bend"
            counts["bend"] += 1
            continue
        between = c.span(p.start - 0.12, p.start + 0.04)
        voiced = between[~np.isnan(between)]
        # a slide: the pitch keeps moving toward the new note for the last ~0.15 s,
        # voiced all the way, and nothing says it was struck again
        gliding = (len(voiced) >= 0.9 * max(1, len(between)) and len(voiced) >= 5
                   and np.all(np.sign(np.diff(voiced)) * np.sign(step) >= -0.05)
                   and abs(voiced[-1] - voiced[0]) >= 0.8 and p.attack < 0.15)
        # a slide is a forced HOPO; other legato notes are left to Clone Hero's own
        # HOPO rule (close notes on different frets), which already covers hammer-ons
        # a hammer-on or pull-off: quick (within an 8th at 120 BPM), no new pick, another note
        legato = p.attack < 0.2 and p.start - q.start <= 0.25
        p.hopo = bool(gliding or legato)
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


THIRTYSECOND = R // 8
OVERTONES = (12, 19, 24, 28, 31)           # semitones above a note where its harmonics ring
LEAD_MIN_NOTES = 4


def without_overtones(pitches: list[int], lead: bool) -> list[int]:
    """Drop pitches that are likely harmonics of a lower one (distortion rings
    octaves and fifths above). In a rhythm part an octave or a fifth above is
    kept: that's how power chords are voiced."""
    ps = sorted(set(pitches))
    harm = OVERTONES if lead else OVERTONES[1:]
    return [x for x in ps if not any(x - y in harm for y in ps if y < x)] or ps[:1]


def is_lead(group: list[Played], ticks: list[int], hi_pitch: float) -> bool:
    """A lead (solo) phrase: mostly one note at a time, and fast or high."""
    if len(group) < LEAD_MIN_NOTES:
        return False
    mono = np.mean([len(without_overtones(p.pitches, True)) == 1 for p in group])
    beats = max(1.0, (ticks[-1] - ticks[0]) / R)
    fast = len(group) / beats >= 2.0                       # 8th notes or quicker
    high = np.median([max(p.pitches) for p in group]) >= hi_pitch
    return mono >= 0.6 and (fast or high)


def plan_frets(pitches: list[int], sizes: list[int], start: int) -> list[int]:
    """The lowest fret of each note in a phrase, by Viterbi over the 5 lanes:
    a step up goes right and down goes left (about a fret per two semitones),
    the same pitch stays on the same fret, and a line that runs out of lanes
    resets to the other side to keep its direction (the Rock Band convention)."""
    n = len(pitches)
    inf = 1e9
    cost = np.full((n, 5), inf)
    back = np.zeros((n, 5), int)
    for f in range(5 - sizes[0] + 1):
        cost[0, f] = 0.4 * abs(f - start)
    for i in range(1, n):
        step = pitches[i] - pitches[i - 1]
        want = int(np.clip(round(abs(step) / 2), 1, 3)) if step else 0
        for f in range(5 - sizes[i] + 1):
            best, arg = inf, 0
            for g in range(5 - sizes[i - 1] + 1):
                if cost[i - 1, g] >= inf:
                    continue
                d = f - g
                edge = (g + sizes[i - 1] - 1 >= 4) if step > 0 else g == 0
                if step == 0:
                    c = 0.0 if d == 0 else 8.0
                elif (step > 0 and d > 0) or (step < 0 and d < 0):
                    c = 0.5 * abs(abs(d) - want)
                elif d == 0:
                    c = 1.5 if edge else 4.0               # equal is fine only against the edge
                else:
                    c = 2.5 if edge else 7.0               # a reset: only when out of room
                c += cost[i - 1, g]
                if c < best:
                    best, arg = c, g
            cost[i, f], back[i, f] = best, arg
    f = int(np.argmin(cost[-1]))
    out = [f]
    for i in range(n - 1, 0, -1):
        f = int(back[i, f])
        out.append(f)
    return out[::-1]


def chart(ps: list[Played], tm, bass: bool = False, max_chord: int = 4) -> tuple[list[Note], dict]:
    """Expert chart notes from the played notes, a phrase at a time with the
    whole song known; and what was done (counts). max_chord: the largest chord,
    in frets (1-4)."""
    items: list[tuple[int, Played]] = []
    for p in sorted(ps, key=lambda p: p.start):
        tick = tm.tick(p.start)
        t = quantize(tick)
        if items and t - items[-1][0] < SIXTEENTH:
            # two starts in one 16th: a quick run (a different note) goes on the 32nd grid
            t32 = int(round(tick / THIRTYSECOND) * THIRTYSECOND)
            if p.pitches != items[-1][1].pitches and t32 - items[-1][0] >= THIRTYSECOND:
                t = t32
            else:
                if p.strength > items[-1][1].strength:
                    items[-1] = (items[-1][0], p)
                continue
        items.append((t, p))
    if not items:
        return [], {}
    ticks = [t for t, _ in items]
    tops = [max(p.pitches) for _, p in items]
    lo, hi = np.percentile(tops, 5), np.percentile(tops, 95)
    hi_pitch = np.percentile(tops, 70)
    # each note: its pitches without overtones, single or chord, and the pitch the line follows
    voiced: list[list[int]] = [[] for _ in items]
    leads = [False] * len(items)
    for ph in phrases(ticks):
        lead = not bass and is_lead([items[i][1] for i in ph], [ticks[i] for i in ph], hi_pitch)
        for i in ph:
            p = items[i][1]
            v = without_overtones(p.pitches, lead)
            # in a solo, a chord only when two or more notes clearly ring together
            if lead and len(v) > 1 and (p.end - p.start < 0.25 or len({x % 12 for x in v}) < 2):
                v = [max(v)]
            voiced[i] = v
            leads[i] = lead
    sizes = [1 if bass else min(CHORD_SIZE[chord_kind(v)], max_chord) for v in voiced]
    line = [max(v) if leads[i] else min(v) for i, v in enumerate(voiced)]
    frets = [0] * len(items)
    seen: dict[tuple, list[int]] = {}
    repeats = 0
    for ph in phrases(ticks):
        key = tuple((ticks[i] - ticks[ph[0]], line[i], sizes[i]) for i in ph)
        if key in seen:                                        # a riff we've charted: the same frets
            for i, f in zip(ph, seen[key]):
                frets[i] = f
            repeats += 1
            continue
        pitches = [line[i] for i in ph]
        if len(set(pitches)) <= 5:
            # the line fits the five lanes: one fret per pitch for the whole phrase
            m = fret_map(pitches, lo, hi)
            plan = [min(m[x], 5 - sizes[i]) for x, i in zip(pitches, ph)]
        else:
            # it doesn't: follow it note by note, resetting when it runs out of room
            center = (np.median(pitches) - lo) / max(1.0, hi - lo)
            plan = plan_frets(pitches, [sizes[i] for i in ph], int(np.clip(round(center * 4), 0, 4)))
        for i, f in zip(ph, plan):
            frets[i] = f
        seen[key] = [frets[i] for i in ph]
    notes: list[Note] = []
    kinds = {"single": 0, "power": 0, "full": 0, "dense": 0}
    for k, (t, p) in enumerate(items):
        kinds["single" if bass else chord_kind(voiced[k])] += 1
        lanes = tuple(range(frets[k], frets[k] + sizes[k]))
        length = int(round(tm.tick(p.end) - t))
        if k + 1 < len(items):
            length = min(length, items[k + 1][0] - t - SUSTAIN_GAP)
        length = (length // SIXTEENTH) * SIXTEENTH
        if p.held is False and p.end - p.start < 0.6:
            length = 0
        if (notes and k and set(p.pitches) <= set(items[k - 1][1].pitches)
                and p.start <= items[k - 1][1].end + HELD_GAP and p.attack < NEW_ATTACK):
            # re-picked while the last note or chord still rings: the same shape again
            lanes = notes[-1].lanes
        forced = None
        if p.hopo and notes:
            forced = "hopo"
            prev = notes[-1]
            step = line[k] - line[k - 1]
            if lanes == prev.lanes and step:
                # a slide or hammer-on lands on another fret, in its direction
                shift = 1 if step > 0 else -1
                if 0 <= lanes[0] + shift and lanes[-1] + shift <= 4:
                    lanes = tuple(x + shift for x in lanes)
        notes.append(Note(t, lanes, length if length >= MIN_SUSTAIN else 0, forced=forced))
    return notes, {"kinds": kinds, "repeated_phrases": repeats, "lead_notes": int(sum(leads)),
                   "lead": list(leads),
                   "gestures": [(n.tick, p.gesture, int(tm.tick(p.end))) for n, (_, p) in zip(notes, items)]}


def extend_sustains(ps: list[Played], c: Contour, floor: float = 0.06) -> list[Played]:
    """Hold a note as long as its pitch keeps sounding: follow the contour past
    the transcriber's end while it stays within a semitone of one of the note's
    pitches (dropouts up to `floor` seconds allowed), up to the next note."""
    ps = sorted(ps, key=lambda p: p.start)
    step = float(c.times[1] - c.times[0]) if len(c.times) > 1 else 0.023
    for k, p in enumerate(ps):
        stop = ps[k + 1].start if k + 1 < len(ps) else p.end + 4.0
        i = int(np.searchsorted(c.times, p.end))
        gap = 0.0
        end = p.end
        while i < len(c.times) and c.times[i] < stop:
            m = c.midi[i]
            if np.isnan(m):
                gap += step
                if gap > floor:
                    break
            elif min(abs(m - x) for x in p.pitches) <= 1.0:
                gap = 0.0
                end = float(c.times[i])
            else:
                break
            i += 1
        p.end = max(p.end, end)
    return ps


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
    """Within each phrase, the share of repeated (root) pitches that land on the
    fret that pitch first had (a chord counts by its lowest fret)."""
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
                same += first[pitch] == notes[i].lanes[0]
            else:
                first[pitch] = notes[i].lanes[0]
    return same / total if total else 1.0
