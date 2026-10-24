"""Note events for guitar and bass: basic-pitch (the opt-in "notes" add-on)
when it's there, else librosa's onsets with pYIN pitch (one note at a time)."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import SR

RANGES = {"guitar": (75.0, 1400.0), "bass": (30.0, 400.0)}
CHORD_WINDOW = 0.045            # note starts this close together are one chord


@dataclass
class Event:
    start: float
    end: float
    pitches: list[int] = field(default_factory=list)      # MIDI, lowest first
    strength: float = 1.0


def basic_pitch_available() -> bool:
    from rhythmsmith_core import addons
    return addons.installed("notes")


def _basic_pitch(path: Path, kind: str, log=print) -> list[list[float]] | None:
    lo, hi = RANGES[kind]
    try:
        from rhythmsmith_core import notes
        import basic_pitch  # noqa: F401
        return notes.transcribe(path, lo, hi)
    except ImportError:
        pass
    from rhythmsmith_core import addons
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "notes.json"
        if not addons.run_script("notes", "notes.py", [path, out, lo, hi], log) or not out.is_file():
            return None
        return json.loads(out.read_text())["notes"]


def group(raw: list[list[float]], min_amp: float = 0.25) -> list[Event]:
    """basic-pitch's notes (start, end, pitch, amplitude) into chords."""
    raw = sorted(r for r in raw if r[3] >= min_amp and r[1] - r[0] >= 0.04)
    out: list[Event] = []
    for s, e, p, a in raw:
        if out and s - out[-1].start <= CHORD_WINDOW:
            ev = out[-1]
            if p not in ev.pitches:
                ev.pitches.append(int(p))
            ev.end = max(ev.end, e)
            ev.strength = max(ev.strength, a)
        else:
            out.append(Event(s, e, [int(p)], a))
    for ev in out:
        ev.pitches.sort()
    return out


def _pyin(y: np.ndarray, sr: int, kind: str) -> list[Event]:
    """One note per onset, pitched by pYIN's median over the note."""
    import librosa
    lo, hi = RANGES[kind]
    hop = 512
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time", backtrack=False)
    if len(onsets) == 0:
        return []
    f0, voiced, _ = librosa.pyin(y, fmin=lo, fmax=min(hi, 1000.0), sr=sr, hop_length=hop,
                                 frame_length=2048 if kind == "guitar" else 4096)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    top = np.percentile(env, 99) or 1.0
    out = []
    for i, s in enumerate(onsets):
        e = onsets[i + 1] if i + 1 < len(onsets) else s + 0.5
        m = (times >= s + 0.02) & (times < e) & voiced
        if m.sum() < 2:
            continue
        p = int(round(librosa.hz_to_midi(np.median(f0[m]))))
        voiced_run = times[(times >= s) & (times < e) & voiced]
        end = float(voiced_run[-1]) if len(voiced_run) else e
        f = min(len(env) - 1, int(s * sr / hop))
        out.append(Event(float(s), max(float(s) + 0.05, end), [p], float(env[f] / top)))
    return out


def events(path: Path | None, y: np.ndarray, kind: str, use_ai: bool = True, sr: int = SR, log=print) -> list[Event]:
    """Note events for "guitar" or "bass" from its stem (file `path`, audio `y`)."""
    if use_ai and path is not None and basic_pitch_available():
        raw = _basic_pitch(path, kind, log)
        if raw:
            return group(raw)
    return _pyin(y, sr, kind)
