"""The arranger: a model that reads a whole song's guitar or bass notes (the
pitches heard at each note start, their timing and length) and gives, per
note, the chord size and HOPO the official charts would use, and whether the
note is part of a solo. It runs with ONNX Runtime on the CPU from the AI
model's folder (arranger_<part>.onnx and .json); without it, the charting
rules decide.

Solo sections come from its per-note solo probability, made sticky: a solo
starts only after a run of notes it is sure of and ends only after a run it
is sure are not, so the chart doesn't flip between solo and rhythm parts.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ai_model

R = 480
NEAR = 0.07
WIN = 128
N_FEAT = 32
SOLO_ON, SOLO_OFF = 0.6, 0.3            # sure it's a solo / sure it isn't (tuned on held-out songs)
SOLO_ENTER, SOLO_LEAVE = 4, 4           # notes in a row needed to start / to end a solo


def model_path(inst: str) -> Path | None:
    p = ai_model.model_dir() / f"arranger_{inst}.onnx"
    if not (p.is_file() and p.with_suffix(".json").is_file()):
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    return p


def available(inst: str) -> bool:
    return model_path(inst) is not None


def lookup(t: float, starts: np.ndarray, evs: list) -> tuple[list[int], float, float]:
    """(pitches, ring length in s, loudness) the transcription has at time t."""
    i0, i1 = np.searchsorted(starts, t - NEAR), np.searchsorted(starts, t + NEAR)
    here = evs[i0:i1] or [e for e in evs[max(0, i0 - 40):i0] if e.start < t < e.end - 0.03]
    if not here:
        return [], 0.0, 0.0
    return sorted({p for e in here for p in e.pitches}), max(e.end for e in here) - t, max(e.strength for e in here)


def features(ticks: list[int], pitch_sets: list[list[int]], rings: list[float], amps: list[float], tm) -> np.ndarray:
    """(n, N_FEAT) per note: pitch content, motion, timing, place in the bar."""
    n = len(ticks)
    f = np.zeros((n, N_FEAT), np.float32)
    tops = [max(p) for p in pitch_sets if p]
    mid = float(np.median(tops)) if tops else 60.0
    prev_lo = prev_hi = None
    for i in range(n):
        ps = pitch_sets[i]
        t = ticks[i]
        gap_prev = (t - ticks[i - 1]) / R if i else 8.0
        gap_next = (ticks[i + 1] - t) / R if i + 1 < n else 8.0
        sec_per_beat = max(1e-3, tm.seconds(t + R) - tm.seconds(t))
        f[i, 0] = 1.0 if ps else 0.0
        f[i, 1] = min(len(ps), 6) / 6
        f[i, 13] = np.log1p(min(gap_prev, 16))
        f[i, 14] = np.log1p(min(gap_next, 16))
        f[i, 15] = np.log1p(rings[i] / sec_per_beat)
        f[i, 16] = amps[i]
        for k, step in enumerate((R * 4, R * 2, R, R // 2, R // 4, R // 3)):
            f[i, 17 + k] = 1.0 if t % step == 0 else 0.0
        f[i, 23], f[i, 24] = np.sin(2 * np.pi * (t % (4 * R)) / (4 * R)), np.cos(2 * np.pi * (t % (4 * R)) / (4 * R))
        if ps:
            lo, hi = min(ps), max(ps)
            f[i, 2] = (lo - mid) / 12
            f[i, 3] = (hi - mid) / 12
            for p in ps:
                f[i, 4 + ((p - lo) % 12) // 2] = 1.0              # the voicing: intervals above the root, by whole tone
            f[i, 10] = min(len({p % 12 for p in ps}), 4) / 4      # distinct pitch classes
            if prev_lo is not None:
                f[i, 25] = np.clip((lo - prev_lo) / 12, -2, 2)
                f[i, 26] = np.clip((hi - prev_hi) / 12, -2, 2)
                f[i, 27] = 1.0 if lo == prev_lo else 0.0
                f[i, 28] = 1.0 if hi == prev_hi else 0.0
            prev_lo, prev_hi = lo, hi
        f[i, 29] = 1.0 if i == 0 or gap_prev >= 1.0 else 0.0           # starts a phrase
        f[i, 30] = (t / R) / 400
        f[i, 31] = 1.0
    return f


@lru_cache(maxsize=4)
def _session(path: str):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def run(inst: str, x: np.ndarray, path: Path | None = None) -> np.ndarray:
    """Per-note outputs (logits): 0-4 lanes, 5-8 chord size 1-4, 9 HOPO, 10 held, 11 solo;
    the song in overlapping windows."""
    sess = _session(str(path or model_path(inst)))
    n = len(x)
    out = np.zeros((n, 12), np.float32)
    cnt = np.zeros(n, np.float32)
    for a in range(0, max(1, n - WIN // 2), WIN // 2):
        seg = x[a:a + WIN]
        padded = np.zeros((WIN, x.shape[1]), np.float32)          # the model takes WIN notes at a time
        padded[: len(seg)] = seg
        o = sess.run(None, {"x": padded[None]})[0][0][: len(seg)]
        out[a:a + len(seg), : o.shape[1]] += o
        cnt[a:a + len(seg)] += 1
    return out / np.maximum(cnt, 1)[:, None]


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def chord_sizes(out: np.ndarray, heard: list[int]) -> list[int]:
    """Frets per note from the chord-size head; never a chord where the
    transcription heard a single pitch."""
    return [1 if h < 2 else int(np.argmax(o[5:9])) + 1 for o, h in zip(out, heard)]


def hopos(out: np.ndarray) -> list[bool]:
    return [bool(sigmoid(o[9]) >= 0.5) for o in out]


def solo_sections(ticks: list[int], out: np.ndarray) -> list[tuple[int, int]]:
    """(start tick, end tick) of each solo: sticky, so a solo runs to its end
    and the chart goes back to the rhythm part only once the solo is over."""
    if out.shape[1] < 12 or not len(ticks):
        return []
    p = sigmoid(out[:, 11])
    spans, start, run_on, run_off = [], None, 0, 0
    for i, v in enumerate(p):
        if start is None:
            run_on = run_on + 1 if v >= SOLO_ON else 0
            if run_on >= SOLO_ENTER:
                start = ticks[i - SOLO_ENTER + 1]
                run_off = 0
        else:
            run_off = run_off + 1 if v <= SOLO_OFF else 0
            if run_off >= SOLO_LEAVE:
                spans.append((start, ticks[i - SOLO_LEAVE + 1]))
                start, run_on = None, 0
    if start is not None:
        spans.append((start, ticks[-1] + R))
    return spans


def meta(inst: str) -> dict:
    p = model_path(inst)
    return json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) if p else {}
