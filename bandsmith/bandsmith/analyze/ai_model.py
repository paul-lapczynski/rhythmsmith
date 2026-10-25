"""The AI model: an optional chart model per instrument, run with ONNX Runtime
on the CPU. Off by default; when it's turned on and a model file is in the
app's data folder (MODEL_DIR/chart_<instrument>.onnx with its .json), it
places that instrument's notes instead of the charting rules. Without one, the
rules are used as always.

It reads the same audio the rules do (the instrument's stem and the full mix)
on the song's own tempo grid, 12 slots per beat, and returns chart notes.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..chart.model import Note

SLOT_TICKS = 40                 # 1/12 beat at 480 ticks per beat
DIFFS = ("easy", "medium", "hard", "expert")


def model_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return Path(os.environ.get("BANDSMITH_MODEL_DIR") or base / "BandSmith" / "models")


def model_path(inst: str) -> Path | None:
    p = model_dir() / f"chart_{inst}.onnx"
    if not (p.is_file() and p.with_suffix(".json").is_file()):
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    return p


def available(inst: str) -> bool:
    return model_path(inst) is not None


@lru_cache(maxsize=4)
def _session(path: str):
    import onnxruntime as ort
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]), meta


def _mel(y: np.ndarray, meta: dict) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=meta["sr"], n_fft=meta["n_fft"], hop_length=meta["hop"],
                                       n_mels=meta["n_mels"], fmin=meta["fmin"], fmax=meta["sr"] / 2)
    return np.log(m + 1e-6).astype(np.float32)


def phase_features(n: int, per_beat: int = 12) -> np.ndarray:
    k = np.arange(n)
    f = np.zeros((n, per_beat + 4), np.float32)
    f[k, k % per_beat] = 1
    bar = (k % (4 * per_beat)) / (4 * per_beat)
    two = (k % (2 * per_beat)) / (2 * per_beat)
    f[:, -4], f[:, -3] = np.sin(2 * np.pi * bar), np.cos(2 * np.pi * bar)
    f[:, -2], f[:, -1] = np.sin(2 * np.pi * two), np.cos(2 * np.pi * two)
    return f


def _windows(mel: np.ndarray, frames: np.ndarray, win: int) -> np.ndarray:
    idx = np.round(frames).astype(int)[:, None] + np.arange(win)[None, :] - win // 2
    return mel[:, np.clip(idx, 0, mel.shape[1] - 1)].transpose(1, 2, 0)        # (S, win, mels)


def _peaks(p: np.ndarray, thr: float, gap: int = 3) -> list[int]:
    keep: list[int] = []
    for i in np.flatnonzero(p >= thr):
        if keep and i - keep[-1] < gap:
            if p[i] > p[keep[-1]]:
                keep[-1] = int(i)
            continue
        keep.append(int(i))
    return keep


def predict(inst: str, stem: np.ndarray, mix: np.ndarray, sr: int, tm, pad: float, end_tick: int,
            diff: str = "expert", path: Path | None = None, lane_threshold: float | None = None,
            chord_threshold: float | None = None) -> list[Note]:
    """Chart notes for `inst` from its stem and the full mix (mono, at `sr`, without
    the lead-in `pad`), on the tempo map `tm` (which includes the pad)."""
    path = path or model_path(inst)
    sess, meta = _session(str(path))
    if sr != meta["sr"]:
        import librosa
        stem = librosa.resample(stem, orig_sr=sr, target_sr=meta["sr"])
        mix = librosa.resample(mix, orig_sr=sr, target_sr=meta["sr"])
    ms, mm = _mel(stem, meta), _mel(mix, meta)
    n = end_tick // SLOT_TICKS + 1
    frames = np.array([(tm.seconds(k * SLOT_TICKS) - pad) * meta["sr"] / meta["hop"] for k in range(n)], np.float32)
    win = meta["win"]
    x = np.stack([_windows(ms, frames, win), _windows(mm, frames, win)], axis=1)[None]     # (1, S, 2, win, mels)
    out = sess.run(None, {"x": x.astype(np.float32), "phase": phase_features(n)[None],
                          "diff": np.array([DIFFS.index(diff)], np.int64)})[0][0]
    on, lanes = out[:, 0], out[:, 1:6]
    thr = meta.get("thresholds", {}).get(diff, 0.5)
    slots = _peaks(on, thr)
    # nothing before the music: slots in the lead-in are silence
    first = int(np.ceil(tm.tick(pad) / SLOT_TICKS))
    slots = [s for s in slots if s >= first]
    lt = lane_threshold if lane_threshold is not None else meta.get("lane_threshold", 0.5)
    notes: list[Note] = []
    for j, s in enumerate(slots):
        lp = lanes[s]
        chosen = [int(i) for i in np.flatnonzero(lp >= lt)] or [int(np.argmax(lp))]
        cymbals: tuple[int, ...] = ()
        length = 0
        if inst == "drums":
            kick = [0] if 0 in chosen else []
            hands = sorted((i for i in chosen if i != 0), key=lambda i: -lp[i])[:2]
            chosen = kick + hands
            cym = out[s, 8:11]
            cymbals = tuple(sorted(i for i in chosen if i >= 2 and cym[i - 2] >= 0.5))
        else:
            # the likeliest fret, plus up to two more when they clear the chord cutoff
            ct = chord_threshold if chord_threshold is not None else meta.get("chord_threshold", lt)
            order = [int(i) for i in np.argsort(-lp)]
            chosen = [order[0]] + [i for i in order[1:3] if lp[i] >= ct]
            sus = float(np.expm1(max(0.0, out[s, 6]))) * SLOT_TICKS
            nxt = slots[j + 1] * SLOT_TICKS if j + 1 < len(slots) else None
            if nxt is not None:
                sus = min(sus, nxt - s * SLOT_TICKS - 120)
            length = int(sus // 120 * 120) if sus >= 240 else 0
        notes.append(Note(s * SLOT_TICKS, tuple(sorted(chosen)), length, cymbals=cymbals))
    return notes
