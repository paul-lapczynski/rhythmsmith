"""A tempo map from beat times.

tempo_map(beats, downbeats) turns the beats a beat tracker found (seconds)
into tempo changes and time signatures at RESOLUTION ticks per beat, so each
beat lands on a whole beat in the chart:

  - before the first beat, a lead-in at the song's opening tempo, so tick 0 is
    the start of the audio;
  - runs of beats that keep one tempo (each beat within `tolerance` seconds of a
    steady grid) share one tempo marker; where the tempo moves, it changes;
  - the downbeats set the bars: 4/4 throughout, with a shorter first bar when
    the music starts mid-bar.

Beats come from detect_beats(): librosa's beat tracker by default, or Beat This!
(an optional download, see rhythmsmith_core.addons) when it's installed and asked for.
"""
from __future__ import annotations

import numpy as np

from .model import RESOLUTION, Tempo, TempoMap, TimeSig

MIN_BPM, MAX_BPM = 40.0, 300.0


def tempo_map(beats, downbeats=None, beats_per_bar: int = 4, tolerance: float = 0.010,
              resolution: int = RESOLUTION) -> tuple[list[Tempo], list[TimeSig], list[float]]:
    """(tempos, time signatures, the time of every beat tick in the map: tick k*resolution
    is at times[k]). Beats are seconds, increasing."""
    b = np.asarray(sorted(set(float(x) for x in beats if x >= 0)), dtype=float)
    if len(b) < 2:
        return [Tempo(0, 120.0)], [TimeSig(0, beats_per_bar, 4)], []
    ibi = float(np.median(np.diff(b[: min(len(b), 9)])))
    # lead-in: whole beats from 0 to the first beat
    lead = int(round(b[0] / ibi)) if b[0] > 0.05 else 0
    if lead == 0 and b[0] > 0.05:
        lead = 1
    times = ([0.0] if lead else []) + list(b)
    anchors = ([0] if lead else []) + [lead + i for i in range(len(b))]     # beat index of each time
    if not lead:
        times[0] = 0.0                     # the first beat is (close enough to) the start
    tempos: list[Tempo] = []
    s = 0
    while s < len(times) - 1:
        e = s + 1
        while e + 1 < len(times):
            # can one tempo cover s..e+1?
            span = (times[e + 1] - times[s]) / (anchors[e + 1] - anchors[s])
            ok = all(abs(times[s] + (anchors[k] - anchors[s]) * span - times[k]) <= tolerance
                     for k in range(s + 1, e + 2))
            if not ok:
                break
            e += 1
        spb = (times[e] - times[s]) / (anchors[e] - anchors[s])
        bpm = 60.0 / spb
        if not (lead and s == 0):          # the lead-in keeps its exact length, whatever its tempo
            bpm = min(MAX_BPM, max(MIN_BPM, bpm))
        tempos.append(Tempo(anchors[s] * resolution, round(bpm, 3)))
        s = e
    # merge neighbours that ended up equal
    merged = [tempos[0]]
    for t in tempos[1:]:
        if abs(t.bpm - merged[-1].bpm) > 1e-3:
            merged.append(t)
    sigs = time_signatures(lead, downbeats, b, beats_per_bar, resolution)
    tm = TempoMap(merged, resolution)
    grid = [tm.seconds(k * resolution) for k in range(anchors[-1] + 1)]
    return merged, sigs, grid


def time_signatures(lead: int, downbeats, beats, beats_per_bar: int, resolution: int) -> list[TimeSig]:
    """4/4 (or beats_per_bar/4) from the first downbeat on; a short pickup bar before it."""
    first_down = 0
    if downbeats is not None and len(downbeats):
        d0 = float(downbeats[0])
        first_down = int(np.argmin(np.abs(np.asarray(beats) - d0)))
    pickup = (lead + first_down) % beats_per_bar
    if pickup == 0:
        return [TimeSig(0, beats_per_bar, 4)]
    return [TimeSig(0, pickup, 4), TimeSig(pickup * resolution, beats_per_bar, 4)]


def downbeat_phase(beats, onset_env, sr_hop: float, beats_per_bar: int = 4) -> list[float]:
    """Guess the downbeats from onset strength (low-frequency onsets, ideally):
    the phase whose beats are strongest. Returns the downbeat times."""
    b = np.asarray(beats)
    if len(b) < beats_per_bar * 2:
        return list(b[:1])
    idx = np.clip(np.round(b / sr_hop).astype(int), 0, len(onset_env) - 1)
    strength = onset_env[idx]
    scores = [strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]
    p = int(np.argmax(scores))
    return list(b[p::beats_per_bar])


def detect_beats(y: np.ndarray, sr: int, low: np.ndarray | None = None) -> tuple[list[float], list[float]]:
    """(beat times, downbeat times) with librosa. `low`, if given, is a signal
    whose onsets mark downbeats well (the drum or bass stem)."""
    import librosa
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, trim=False)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if low is not None:
        lenv = librosa.onset.onset_strength(y=low, sr=sr, hop_length=hop, fmax=200)
    else:
        lenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200)
    downs = downbeat_phase(beats, lenv, hop / sr)
    return [float(t) for t in beats], [float(t) for t in downs]


def beat_this_beats(audio, log=print) -> tuple[list[float], list[float]] | None:
    """(beats, downbeats) from Beat This!, if its add-on is installed; else None."""
    import json
    import tempfile
    from pathlib import Path

    from rhythmsmith_core import addons
    if not addons.installed("beats"):
        return None
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "beats.json"
        if not addons.run_script("beats", "beats.py", [audio, out], log) or not out.is_file():
            return None
        r = json.loads(out.read_text())
    return r["beats"], r["downbeats"]
