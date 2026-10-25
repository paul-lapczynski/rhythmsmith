"""Vocal notes: pYIN pitch on the vocal stem, cut into notes where the pitch
moves; with Whisper word timings (the opt-in lyrics add-on) the words become
syllables on those notes. Without Whisper, notes get a placeholder syllable."""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR_V = 16000
HOP_V = 320                 # 20 ms
MIN_NOTE = 0.08             # seconds
PLACEHOLDER = "ah"


@dataclass
class VNote:
    start: float
    end: float
    pitch: int
    lyric: str = PLACEHOLDER


def pitch_track(y16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, MIDI pitch or nan) every 20 ms."""
    import librosa
    f0, voiced, _ = librosa.pyin(y16, fmin=80, fmax=1000, sr=SR_V, hop_length=HOP_V, frame_length=1024)
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    return librosa.times_like(f0, sr=SR_V, hop_length=HOP_V), midi


def segment(times: np.ndarray, midi: np.ndarray) -> list[VNote]:
    """Voiced runs, split where the pitch settles a semitone or more away."""
    out: list[VNote] = []
    i, n = 0, len(midi)
    step = times[1] - times[0] if n > 1 else 0.02
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        cur = [midi[i]]
        while j + 1 < n and not np.isnan(midi[j + 1]):
            ref = np.median(cur[-8:])
            ahead = midi[j + 1:j + 4]
            ahead = ahead[~np.isnan(ahead)]
            if len(ahead) >= 2 and abs(np.median(ahead) - ref) >= 0.9 and len(cur) * step >= MIN_NOTE:
                break
            j += 1
            cur.append(midi[j])
        s, e = times[i], times[j] + step
        if e - s >= MIN_NOTE:
            p = int(round(np.median(cur)))
            while p < 36:
                p += 12
            while p > 84:
                p -= 12
            out.append(VNote(float(s), float(e), p))
        i = j + 1
    return out


def syllables(word: str) -> list[str]:
    """A rough split of a word into syllables at vowel groups ("singing" -> sing, ing)."""
    w = re.sub(r"[^\w']", "", word)
    if len(w) <= 3:
        return [w] if w else []
    parts = re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy]*$|[^aeiouy](?=[^aeiouy]))?", w, flags=re.I)
    if not parts or "".join(parts) != w:
        return [w]
    if len(parts) > 1 and re.fullmatch(r"[^aeiouy]*e", parts[-1], flags=re.I):   # silent e
        last = parts.pop()
        parts[-1] += last
    return parts


def whisper_words(vocals: Path, log=print) -> list[list] | None:
    from rhythmsmith_core import addons
    if not addons.installed("lyrics"):
        return None
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "words.json"
        if not addons.transcribe(vocals, out, log) or not out.is_file():
            return None
        return json.loads(out.read_text())["words"]


def attach_lyrics(notes: list[VNote], words: list[list]) -> list[VNote]:
    """Each word's syllables on the notes it covers; a word's extra notes are
    "+" (the same syllable carried on); notes under no word are dropped."""
    out: list[VNote] = []
    for ws, we, word in words:
        under = [n for n in notes if n.end > ws and n.start < we and min(n.end, we) - max(n.start, ws) > 0.03]
        syl = syllables(word)
        if not under or not syl:
            continue
        if len(syl) > len(under):
            syl = syl[: len(under) - 1] + ["".join(syl[len(under) - 1:])]
        for k, n in enumerate(under):
            if k < len(syl):
                n.lyric = syl[k] + ("-" if k < len(syl) - 1 else "")
            else:
                n.lyric = "+"
            out.append(n)
    seen, uniq = set(), []
    for n in sorted(out, key=lambda n: n.start):
        if id(n) not in seen:
            seen.add(id(n))
            uniq.append(n)
    return uniq


def notes(vocals_path: Path, y16: np.ndarray, use_whisper: bool = True, log=print) -> tuple[list[VNote], bool]:
    """(notes, whether they have real lyrics)."""
    times, midi = pitch_track(y16)
    ns = segment(times, midi)
    words = whisper_words(vocals_path, log) if use_whisper else None
    if words:
        return attach_lyrics(ns, words), True
    # no words: keep the longer, clearer notes only
    return [n for n in ns if n.end - n.start >= 0.15], False
