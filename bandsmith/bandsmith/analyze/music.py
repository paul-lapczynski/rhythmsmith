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
    chord_ok: bool | None = None  # the AI model's call on whether a chord belongs here (None: no model)
    amp: float = 1.0              # how loud the transcription heard it (0-1)
    slide_in: tuple | None = None  # (glide start s, from pitch) when it slides into its pitch
    slide_out: tuple | None = None  # (glide end s, to pitch) when it slides off at its end


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
                near: float = 0.07, held=None, sure: float = 0.6, held_cut: float = 0.35,
                chordy=None, chord_cut: float = 0.3) -> list[Played]:
    """A note at each onset, with the pitches the transcription hears there: the
    ones starting within `near` seconds (a chord if several), else the ones still
    ringing (a re-struck note), else the contour's pitch. With no pitch heard, a
    sure onset (strength >= `sure`, e.g. a palm-muted chug) repeats the last
    note's pitches; otherwise no note. `held`: the AI model's sustained-or-not
    probability per onset, if it has one; `chordy`: its probability of a chord there
    (two frets or more), which can only veto a chord the transcription heard."""
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
        amp = 0.0
        if here:
            pitches = sorted({p for e in here for p in e.pitches})
            end = max(e.end for e in here)
            amp = max(e.strength for e in here)
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
        p.amp = float(amp)
        if held is not None:
            p.held = bool(held[order[k]] >= held_cut)
        if chordy is not None:
            p.chord_ok = bool(chordy[order[k]] >= chord_cut)
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
        # only slides are forced: hammer-ons and pull-offs are left to Clone Hero's own
        # HOPO rule (a quick note on another fret), which matches the official charts better
        p.hopo = bool(gliding)
        p.gesture = "slide" if gliding else "legato"
        counts[p.gesture] += 1
        q.end = min(q.end, p.start) if p.gesture == "slide" else q.end
        out.append(p)
    return out, counts


# ------------------------------------------------------------------ slides into and off notes

GLIDE_MIN, GLIDE_MAX = 0.04, 0.25      # seconds a slide takes
GLIDE_STEP = 1.0                       # semitones it covers at least


def _glide(seg: np.ndarray, step_s: float) -> tuple[int, int] | None:
    """(start, end) frame of a steady glide in a contour segment: voiced, moving one
    way, covering GLIDE_STEP or more within GLIDE_MIN-GLIDE_MAX seconds."""
    voiced = np.flatnonzero(~np.isnan(seg))
    if len(voiced) < 3:
        return None
    seg = seg[voiced[0]:voiced[-1] + 1]                # silence before or after doesn't count
    if np.isnan(seg).mean() > 0.2:
        return None
    s = seg[~np.isnan(seg)]
    d = np.diff(s)
    total = s[-1] - s[0]
    if abs(total) < GLIDE_STEP or not np.all(np.sign(d) * np.sign(total) >= -0.1):
        return None
    dur = len(s) * step_s
    return (0, len(s)) if GLIDE_MIN <= dur <= GLIDE_MAX + step_s else None


def slides(ps: list[Played], c: Contour) -> dict:
    """Mark slides into notes (the pitch glides up or down to the note's pitch just
    before or at its start) and off them (it glides away at the end of a held
    note). Returns counts."""
    step_s = float(c.times[1] - c.times[0]) if len(c.times) > 1 else 0.023
    n_in = n_out = 0
    ps = sorted(ps, key=lambda p: p.start)
    for k, p in enumerate(ps):
        target = p.pitches[-1] if len(p.pitches) == 1 else None
        if target is None:
            continue
        prev_end = ps[k - 1].end if k else 0.0
        i0 = int(np.searchsorted(c.times, max(prev_end, p.start - GLIDE_MAX)))
        i1 = int(np.searchsorted(c.times, p.start + 0.06))
        seg = c.midi[i0:i1]
        if _glide(seg, step_s) and abs(np.nanmedian(seg[-2:]) - target) <= 0.6:
            first = seg[~np.isnan(seg)][0]
            p.slide_in = (float(c.times[i0]), int(round(first)))
            n_in += 1
        nxt = ps[k + 1].start if k + 1 < len(ps) else p.end + GLIDE_MAX
        j0 = int(np.searchsorted(c.times, p.end - 0.03))
        j1 = int(np.searchsorted(c.times, min(nxt, p.end + GLIDE_MAX)))
        seg = c.midi[j0:j1]
        if p.end - p.start >= 0.25 and _glide(seg, step_s) and abs(np.nanmedian(seg[:2]) - target) <= 0.6:
            last = seg[~np.isnan(seg)][-1]
            p.slide_out = (float(c.times[min(j1, len(c.times) - 1)]), int(round(last)))
            n_out += 1
    return {"slide_in": n_in, "slide_out": n_out}


def legato_notes(ps: list[Played], c: Contour, near: float = 0.06, hold: float = 0.06,
                 busy: float = 0.3, events: list[Event] | None = None) -> list[Played]:
    """Quick notes played without a new pick (hammer-ons, pull-offs, fast runs):
    wherever the stem's pitch moves to a new semitone and holds it for `hold`
    seconds or more, inside a busy passage (notes within `busy` seconds on both
    sides), and no note starts there yet. Each needs its pitch heard (and, given
    the transcription's `events`, a transcribed note of that pitch starting there
    too): nothing is invented."""
    if not ps or len(c.times) < 3:
        return []
    step_s = float(c.times[1] - c.times[0])
    need = max(2, int(np.ceil(hold / step_s)))
    starts = np.array(sorted(p.start for p in ps))
    m = np.round(c.midi)
    out: list[Played] = []
    i = 1
    while i < len(m):
        if np.isnan(m[i]) or np.isnan(m[i - 1]) or m[i] == m[i - 1]:
            i += 1
            continue
        j = i
        while j + 1 < len(m) and m[j + 1] == m[i]:
            j += 1
        t0 = float(c.times[i])
        k = int(np.searchsorted(starts, t0))
        before = starts[k - 1] if k else -1e9
        after = starts[k] if k < len(starts) else 1e9
        steady = np.nanstd(c.midi[i:j + 1]) < 0.15         # a held pitch, not part of a slide
        if steady and events is not None:
            steady = any(abs(e.start - c.times[i]) <= near and int(m[i]) in e.pitches for e in events
                         if abs(e.start - c.times[i]) <= near)
        if j - i + 1 >= need and steady and t0 - before >= near and after - t0 >= near \
                and t0 - before <= busy and after - t0 <= busy:
            out.append(Played(t0, float(c.times[j]) + step_s, [int(m[i])], 0.5, 0.0))
            out[-1].gesture = "legato"
        i = j + 1
    return out


# ------------------------------------------------------------------ lead and rhythm voices

VOICES = ("left", "center", "right")
SOLO_SHARE = float(__import__("os").environ.get("BANDSMITH_SOLO_SHARE", "0.6"))   # the lead voice's share of the loudness
SOLO_MIN = int(__import__("os").environ.get("BANDSMITH_SOLO_MIN", "4"))            # windows (of two beats) in a solo
SOLO_WINDOW = 2 * R          # solos are found two beats at a time
VOICE_HOP = 512


def stereo_voices(y2: np.ndarray, sr: int) -> dict[str, np.ndarray] | None:
    """The guitar stem split by where each sound sits in the stereo field (left,
    center, right), as mono signals; None for a mono stem. Two guitars panned
    apart (a lead over a rhythm part, say) come out separately."""
    import librosa
    if y2.ndim != 2 or y2.shape[1] < 2:
        return None
    left, right = y2[:, 0], y2[:, 1]
    if np.corrcoef(left[:: 50], right[:: 50])[0, 1] > 0.98:            # really mono
        return None
    sl = librosa.stft(left, n_fft=2048, hop_length=VOICE_HOP)
    sr_ = librosa.stft(right, n_fft=2048, hop_length=VOICE_HOP)
    a, b = np.abs(sl), np.abs(sr_)
    pan = a / (a + b + 1e-9)                                           # 1: all left, 0: all right
    mid = (sl + sr_) / 2
    masks = {"left": np.clip((pan - 0.55) / 0.15, 0, 1), "right": np.clip((0.45 - pan) / 0.15, 0, 1)}
    masks["center"] = 1 - masks["left"] - masks["right"]
    return {k: librosa.istft(mid * m, hop_length=VOICE_HOP, length=len(left)).astype(np.float32)
            for k, m in masks.items()}


def loudness(y: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.feature.rms(y=y, hop_length=VOICE_HOP)[0]


def solo_spans(voice_events: dict[str, list[Event]], voice_rms: dict[str, np.ndarray], sr: int, tm,
               offset: float, first: int, last: int) -> list[tuple[int, int, str]]:
    """(start tick, end tick, lead voice) of each solo: two-beat windows where one
    voice carries most of the guitar's loudness (SOLO_SHARE) and plays a single
    line (one pitch at a time, at least three notes), joined into sections of a
    bar or more. Speed or register alone don't make a solo."""
    frames_per_s = sr / VOICE_HOP
    wins = []
    t = (first // SOLO_WINDOW) * SOLO_WINDOW
    while t < last:
        a, b = tm.seconds(t) - offset, tm.seconds(t + SOLO_WINDOW) - offset
        fa, fb = int(a * frames_per_s), max(int(a * frames_per_s) + 1, int(b * frames_per_s))
        energy = {v: float(np.mean(voice_rms[v][fa:fb] ** 2)) if fb <= len(voice_rms[v]) else 0.0 for v in voice_rms}
        total = sum(energy.values()) or 1.0
        lead, share = max(((v, e / total) for v, e in energy.items()), key=lambda x: x[1])
        evs = [e for e in voice_events.get(lead, []) if a <= e.start - offset < b]
        single = np.mean([len(without_overtones(e.pitches, True)) == 1 for e in evs]) if evs else 0.0
        moving = len({max(e.pitches) for e in evs}) >= 3
        wins.append((t, lead if (share >= SOLO_SHARE and len(evs) >= 4 and single >= 0.8 and moving) else None))
        t += SOLO_WINDOW
    spans: list[list] = []
    for t, lead in wins:
        if lead and spans and spans[-1][1] == t and spans[-1][2] == lead:
            spans[-1][1] = t + SOLO_WINDOW
        elif lead:
            spans.append([t, t + SOLO_WINDOW, lead])
    return [(a, b, v) for a, b, v in spans if b - a >= SOLO_MIN * SOLO_WINDOW]


PAN_SHARE = 0.55            # a side voice's share of the loudness that makes a solo
PAN_QUIET = 0.4             # ...and its usual (median) share, for it to count as a lead that comes and goes
PAN_SMOOTH_BARS = 4


PAN_MONO = 0.7             # the lead side plays one note at a time at least this often
PAN_SWING = 0.25            # ...and its share swings by this much (95th percentile over median)


def pan_solos(voice_rms: dict[str, np.ndarray], sr: int, tm, offset: float, first: int, last: int,
              voice_events: dict[str, list[Event]] | None = None, min_bars: int = 2) -> list[tuple[int, int, str]]:
    """Solos from where the guitar sits in the stereo mix: one side voice (left or
    right), the one that plays single notes most (a lead line), usually in the
    background but taking over for two bars or more, smoothed over four bars.
    Nothing for songs without such a lead (no side that mostly plays single
    notes, or whose loudness doesn't swing)."""
    bar = 4 * R
    frames_per_s = sr / VOICE_HOP
    starts = list(range((first // bar) * bar, last, bar))
    shares = {v: [] for v in ("left", "right")}
    for t in starts:
        a, b = tm.seconds(t) - offset, tm.seconds(t + bar) - offset
        fa, fb = int(a * frames_per_s), max(int(a * frames_per_s) + 1, int(b * frames_per_s))
        e = {v: float(np.mean(voice_rms[v][fa:fb] ** 2)) if fa < len(voice_rms[v]) else 0.0 for v in voice_rms}
        total = sum(e.values()) or 1.0
        for v in shares:
            shares[v].append(e.get(v, 0.0) / total)
    out: list[tuple[int, int, str]] = []
    k = np.ones(PAN_SMOOTH_BARS) / PAN_SMOOTH_BARS
    sides = list(shares)
    if voice_events:
        mono = {v: np.mean([len(without_overtones(e.pitches, True)) == 1 for e in voice_events.get(v, [])] or [0])
                for v in sides}
        lead = max(sides, key=lambda v: mono[v])
        sides = [lead] if mono[lead] >= PAN_MONO else []
    for v in sides:
        s = np.array(shares[v])
        if len(s) < PAN_SMOOTH_BARS or np.median(s) > PAN_QUIET or np.percentile(s, 95) - np.median(s) < PAN_SWING:
            continue                                   # always loud, or never stands out: not a lead
        smooth = np.convolve(s, k, mode="same")
        on = smooth >= PAN_SHARE
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                if j - i + 1 >= min_bars:
                    out.append((starts[i], starts[j] + bar, v))
                i = j + 1
            else:
                i += 1
    return sorted(out)


def solo_notes(events: list[Event], span: tuple[int, int], tm, offset_ok: float = 0.0,
               min_amp: float = 0.3) -> list[Played]:
    """The lead voice's own notes inside a solo (each with its transcribed pitch:
    nothing without pitch evidence)."""
    a, b = tm.seconds(span[0]), tm.seconds(span[1])
    out = []
    for e in events:
        if a <= e.start < b and e.strength >= min_amp:
            p = Played(e.start, e.end, [max(without_overtones(e.pitches, True))], e.strength, 1.0)
            p.amp = e.strength
            out.append(p)
    return out


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
LEAD_MIN_NOTES = 6
FAINT = 0.4                               # transcription loudness below which a chord is faint


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
    fast = len(group) / beats >= 3.0                       # quicker than 8th notes
    high = np.median([max(p.pitches) for p in group]) >= hi_pitch
    moving = len({max(p.pitches) for p in group}) >= 4     # a line, not one riff note repeated
    return bool(mono >= 0.8 and moving and (fast or high))


def plan_frets(pitches: list[int], sizes: list[int], start: int, prefer: dict | None = None) -> list[int]:
    """The lowest fret of each note in a phrase, by Viterbi over the 5 lanes:
    a step up goes right and down goes left (about a fret per two semitones),
    the same pitch stays on the same fret, and a line that runs out of lanes
    resets to the other side to keep its direction (the Rock Band convention).
    `prefer`: a fret per pitch to stay on where the line allows (same pitch, same fret)."""
    n = len(pitches)
    inf = 1e9
    cost = np.full((n, 5), inf)
    back = np.zeros((n, 5), int)
    def keep(i, f):
        return 4.0 if prefer and pitches[i] in prefer and prefer[pitches[i]] != f else 0.0
    for f in range(5 - sizes[0] + 1):
        cost[0, f] = 0.4 * abs(f - start) + keep(0, f)
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
                c += cost[i - 1, g] + keep(i, f)
                if c < best:
                    best, arg = c, g
            cost[i, f], back[i, f] = best, arg
    f = int(np.argmin(cost[-1]))
    out = [f]
    for i in range(n - 1, 0, -1):
        f = int(back[i, f])
        out.append(f)
    return out[::-1]


def chart(ps: list[Played], tm, bass: bool = False, max_chord: int = 4,
          strict: float | None = None, solos: list[tuple[int, int, str]] | None = None,
          arranged: dict[int, tuple[int, bool]] | None = None) -> tuple[list[Note], dict]:
    """Expert chart notes from the played notes, a phrase at a time with the
    whole song known; and what was done (counts). max_chord: the largest chord,
    in frets (1-4). strict: notes found with a lower bar (strength under
    `strict`) are kept only in solos, where quick quiet notes matter. solos: the
    solo sections (solo_spans), when the voices could be told apart; without
    them a phrase is a solo by how it moves (is_lead). arranged: the arranger
    model's (chord size, HOPO) by note tick; the frets, sustains and the
    same-note rule stay the rules'."""
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
        if solos is not None:
            lead = not bass and any(a <= ticks[i] < b for i in ph for a, b, _ in solos)
        else:
            lead = not bass and is_lead([items[i][1] for i in ph], [ticks[i] for i in ph], hi_pitch)
        for i in ph:
            if solos is not None and lead:
                leads[i] = any(a <= ticks[i] < b for a, b, _ in solos)
                if not leads[i]:
                    voiced[i] = without_overtones(items[i][1].pitches, False)
                    continue
            p = items[i][1]
            v = without_overtones(p.pitches, lead)
            # in a solo, single notes, unless a power chord is clearly held
            if lead and len(v) > 1 and not (chord_kind(v) == "power" and p.end - p.start >= 0.4):
                v = [max(v)]
            # pitches heard together, but the AI model hears no chord here: in a solo, or
            # when the chord is faint, it's likely bleed or let-ring; a clear rhythm chord stays
            if len(v) > 1 and p.chord_ok is False and (lead or p.amp < FAINT):
                v = [max(v)] if lead else [min(v)]
            voiced[i] = v
            leads[i] = lead
    if strict is not None:
        keep = [k for k in range(len(items)) if leads[k] or items[k][1].strength >= strict]
        if len(keep) < len(items):
            items = [items[k] for k in keep]
            voiced = [voiced[k] for k in keep]
            leads = [leads[k] for k in keep]
            ticks = [t for t, _ in items]
    sizes = [1 if bass else min(CHORD_SIZE[chord_kind(v)], max_chord) for v in voiced]
    if arranged and not bass:
        # the arranger's chord sizes (never a chord where one pitch is heard; in a solo, never bigger)
        for i, t in enumerate(ticks):
            if t in arranged:
                size = min(arranged[t][0], max_chord)
                sizes[i] = min(size, sizes[i]) if leads[i] else (size if len(voiced[i]) > 1 else 1)
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
            sz, st = [sizes[i] for i in ph], int(np.clip(round(center * 4), 0, 4))
            plan = plan_frets(pitches, sz, st)
            # again, now preferring each pitch's most used fret: repeats stay put between resets
            usual = {x: max(set(f for y, f in zip(pitches, plan) if y == x),
                            key=lambda f: sum(1 for y, g in zip(pitches, plan) if y == x and g == f))
                     for x in set(pitches)}
            plan = plan_frets(pitches, sz, st, usual)
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
            room = items[k + 1][0] - t - SUSTAIN_GAP
            # a note that rings through most of the space to the next one is held up to it,
            # as the official charts do
            if room >= MIN_SUSTAIN and length >= HOLD_FILL * room:
                length = room
            length = min(length, room)
        length = (length // SIXTEENTH) * SIXTEENTH
        if HELD_VETO and p.held is False and p.end - p.start < 0.4:
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
        if notes and voiced[k] == voiced[k - 1] and line[k] == line[k - 1]:
            # the same note or chord again: always the same frets (a hard rule, across phrases)
            lanes = notes[-1].lanes
        if arranged and forced is None and t in arranged and arranged[t][1] is not None:
            # HOPO or strum as the arranger says, where it differs from Clone Hero's own rule
            natural = bool(notes) and len(lanes) == 1 and t - notes[-1].tick <= 170 and lanes != notes[-1].lanes
            if arranged[t][1] and not natural:
                forced = "hopo"
            elif not arranged[t][1] and natural:
                forced = "strum"
        notes.append(Note(t, lanes, length if length >= MIN_SUSTAIN else 0, forced=forced))
    notes, slide_counts = slide_notes(notes, items, line, tm, bass)
    return notes, {"kinds": kinds, "repeated_phrases": repeats, "lead_notes": int(sum(leads)), **slide_counts,
                   "lead": list(leads), "voiced": {it[0]: tuple(v) for it, v in zip(items, voiced)},
                   "gestures": [(n.tick, p.gesture, int(tm.tick(p.end))) for n, (_, p) in zip(notes, items)]}


GRACE_GAP = 0.05             # seconds: how far ahead of its target a slide's lead-in note sits
HOLD_FILL = float(__import__("os").environ.get("BANDSMITH_HOLD_FILL", "0.6"))   # ring share of the gap that holds to the next note
HELD_VETO = __import__("os").environ.get("BANDSMITH_HELD_VETO", "0") == "1"      # let the AI model cut a short ring (it held too few)


def slide_notes(notes: list[Note], items, line, tm, bass: bool) -> tuple[list[Note], dict]:
    """Slides as the official charts write them: a slide into a note gets a short
    lead-in note a step below (or above) just before the target, which becomes a
    HOPO (always on bass, where charts do it about half the time; on guitar only
    for slides of two semitones or more); a slide off a held note gets a short
    HOPO note where it lands, and the held note ends before it."""
    out = list(notes)
    taken = sorted(n.tick for n in notes)
    import bisect

    def free(tick):
        i = bisect.bisect_left(taken, tick)
        return all(abs(tick - taken[j]) >= THIRTYSECOND for j in (i - 1, i) if 0 <= j < len(taken))

    n_in = n_out = 0
    for k, (t, p) in enumerate(items):
        n = notes[k]
        if len(n.lanes) != 1:
            continue
        if p.slide_in:
            frm = p.slide_in[1]
            step = line[k] - frm
            if step and (bass or abs(step) >= 2):
                g = int(round(tm.tick(p.start - GRACE_GAP) / THIRTYSECOND) * THIRTYSECOND)
                f = n.lanes[0] - (1 if step > 0 else -1)
                if t - g >= THIRTYSECOND and 0 <= f <= 4 and free(g):
                    out.append(Note(g, (f,)))                  # appended: out[k] is still this note
                    bisect.insort(taken, g)
                    n = out[k] = Note(n.tick, n.lanes, n.length, forced="hopo")
                    n_in += 1
        if p.slide_out and n.length:
            to = p.slide_out[1]
            step = to - line[k]
            g = int(round(tm.tick(p.slide_out[0]) / THIRTYSECOND) * THIRTYSECOND)
            f = n.lanes[0] + (1 if step > 0 else -1)
            if step and g - t >= MIN_SUSTAIN and 0 <= f <= 4 and free(g):
                length = g - t - SIXTEENTH
                out[k] = Note(n.tick, n.lanes, length if length >= MIN_SUSTAIN else 0, forced=n.forced)
                out.append(Note(g, (f,), forced="hopo"))
                bisect.insort(taken, g)
                n_out += 1
    out.sort(key=lambda n: n.tick)
    return out, {"slide_in_notes": n_in, "slide_out_notes": n_out}


def extend_sustains(ps: list[Played], c: Contour, floor: float = 0.12) -> list[Played]:
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

def repeat_switches(notes: list[Note], voiced: dict[int, tuple]) -> int:
    """Consecutive chart notes with the same pitch (or voicing) on different frets:
    must be 0 (a real instrument doesn't move to play the same note again).
    `voiced`: tick -> pitches, from chart()'s info."""
    return sum(1 for a, b in zip(notes, notes[1:])
               if a.tick in voiced and b.tick in voiced and voiced[a.tick] == voiced[b.tick] and a.lanes != b.lanes)


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
