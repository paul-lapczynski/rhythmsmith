"""Drum hits from a drum stem: onsets in frequency bands, sorted into kick,
snare, hi-hat and cymbals, and toms.

    kick       onsets in 30-110 Hz
    snare      a noisy onset in 1.5-5 kHz with body in 150-400 Hz
    cymbals    onsets in 7-11 kHz; short ones are the hi-hat (yellow cymbal),
               long-ringing ones a crash (green cymbal)
    toms       a tonal low-mid onset (70-350 Hz) that isn't the kick or snare;
               high, mid and floor tom by pitch (yellow, blue, green)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR

HOP = 256
N_FFT = 1024


@dataclass
class Hit:
    time: float
    lane: int                   # 0 kick, 1 snare, 2 yellow, 3 blue, 4 green
    cymbal: bool
    strength: float


def _band(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return S[(freqs >= lo) & (freqs < hi)]


def _env(S_band: np.ndarray) -> np.ndarray:
    """Onset strength of a band: positive change of its log energy, normalized."""
    e = np.log1p(100 * S_band.sum(axis=0))
    d = np.maximum(0, np.diff(e, prepend=e[:1]))
    top = np.percentile(d, 99.5) or 1.0
    return d / top


def _peaks(env: np.ndarray, delta: float, wait: int = 4) -> np.ndarray:
    import librosa
    return librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=12, post_avg=12, delta=delta, wait=wait)


def hits(y: np.ndarray, sr: int = SR, sensitivity: float = 1.0) -> list[Hit]:
    """Every onset of the drum stem, then what it is from how much each band
    jumps at it: several drums can sound at once (kick and hi-hat, say)."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    bands = {"low": (30, 110), "body": (150, 400), "snap": (1500, 5000), "tom": (70, 350), "high": (7000, 11000)}
    energy = {k: _band(S, freqs, lo, hi).sum(axis=0) + 1e-10 for k, (lo, hi) in bands.items()}
    envs = {k: _env(_band(S, freqs, lo, hi)) for k, (lo, hi) in bands.items()}
    total = envs["low"] + envs["snap"] + envs["high"] + 0.5 * envs["tom"]
    onsets = _peaks(total / (np.percentile(total, 99.5) or 1.0), 0.12 / sensitivity, wait=3)
    n = len(t)

    def rise(k, f):
        """the band's jump at frame f: its strongest onset strength nearby"""
        return float(envs[k][max(0, f - 1):min(n, f + 3)].max())

    out: list[Hit] = []
    cf = freqs[(freqs >= 70) & (freqs < 350)]
    tomband = _band(S, freqs, 70, 350)
    for f in onsets:
        lo, bo, sn, hi, to = (rise(k, f) for k in ("low", "body", "snap", "high", "tom"))
        at = min(n - 1, f + 1)
        kick = lo > 0.30
        # a snare: the 1.5-5 kHz crack rises at least as much as the top end, with body under it
        snare = sn > 0.30 and sn >= 0.8 * hi and bo > 0.20 and energy["snap"][at] > 0.5 * energy["high"][at]
        if kick:
            out.append(Hit(t[f], 0, False, lo))
        if snare:
            out.append(Hit(t[f], 1, False, sn))
        if hi > 0.30 and not (snare and hi < 1.2 * sn):
            # how long the top end rings: a hi-hat dies within ~80 ms, a crash keeps going
            later = energy["high"][min(n - 1, f + 1 + int(0.08 * sr / HOP))]
            ring = later / energy["high"][at]
            out.append(Hit(t[f], 4 if ring > 0.5 else 2, True, hi))
        if to > 0.40 and not kick and not snare and lo < 0.6 * to:
            frame = tomband[:, min(n - 1, f + 2)]
            c = float((frame * cf).sum() / (frame.sum() + 1e-12))
            out.append(Hit(t[f], 2 if c > 180 else 3 if c > 120 else 4, False, to))
    return sorted(out, key=lambda h: h.time)


def to_chords(hs: list[Hit], window: float = 0.03) -> list[list[Hit]]:
    """Hits within `window` of each other, together; at most one per lane (the strongest)."""
    groups: list[list[Hit]] = []
    for h in hs:
        if groups and h.time - groups[-1][0].time <= window:
            groups[-1].append(h)
        else:
            groups.append([h])
    out = []
    for g in groups:
        best: dict[int, Hit] = {}
        for h in g:
            if h.lane not in best or h.strength > best[h.lane].strength:
                best[h.lane] = h
        out.append(sorted(best.values(), key=lambda h: h.lane))
    return out


def limb_limit(chord: list[Hit]) -> list[Hit]:
    """Two hands and a foot: the kick, plus the two strongest of the rest."""
    kick = [h for h in chord if h.lane == 0]
    hands = sorted((h for h in chord if h.lane != 0), key=lambda h: -h.strength)[:2]
    return sorted(kick + hands, key=lambda h: h.lane)
