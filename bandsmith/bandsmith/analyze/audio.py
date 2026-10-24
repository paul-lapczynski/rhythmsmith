"""Reading and writing audio, and splitting a song into stems."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

SR = 22050                  # analysis rate
OUT_SR = 44100              # what the song folder's .ogg files are written at
STEM_NAMES = ("vocals", "drums", "bass", "guitar", "piano", "other")


def load(path: Path, sr: int = SR, mono: bool = True) -> np.ndarray:
    """The file as float32 at `sr`: (samples,) mono, or (samples, 2)."""
    import soundfile as sf
    try:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:                       # formats libsndfile can't read: librosa's fallbacks
        import librosa
        y, file_sr = librosa.load(str(path), sr=None, mono=False)
        y = np.atleast_2d(y).T.astype(np.float32)
    if mono:
        y = y.mean(axis=1)
        if file_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        return y.astype(np.float32)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y[:, :2]
    if file_sr != sr:
        import librosa
        y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr).T
    return np.ascontiguousarray(y, dtype=np.float32)


def write_ogg(path: Path, y: np.ndarray, sr: int = OUT_SR) -> None:
    import soundfile as sf
    y = np.clip(np.atleast_2d(y.T).T if y.ndim == 1 else y, -1, 1).astype(np.float32)
    ch = 1 if y.ndim == 1 else y.shape[1]
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=ch, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(y), sr * 10):            # in blocks: big Vorbis writes can crash
            f.write(y[i:i + sr * 10])


def write_padded(src: Path, dest: Path, pad: float) -> None:
    """Copy an audio file to an .ogg with `pad` seconds of silence in front."""
    if pad <= 0 and src.suffix.lower() == ".ogg":
        shutil.copyfile(src, dest)
        return
    y = load(src, OUT_SR, mono=False)
    y = np.concatenate([np.zeros((int(round(pad * OUT_SR)), 2), np.float32), y])
    write_ogg(dest, y)


def stems(audio: Path, cache: Path, guitar: bool = True, given: Path | None = None, log=print) -> dict[str, Path]:
    """Stem files for a song, by name (vocals, drums, bass, guitar, piano, other).

    `given`: a folder of stems already separated (vocals.ogg, ...). Otherwise
    Demucs through the vocal separation add-on, htdemucs_6s when guitar is
    wanted (it has a guitar stem), into `cache`. Empty when neither is there."""
    if given is not None:
        return {n: given / f"{n}.ogg" for n in STEM_NAMES if (given / f"{n}.ogg").is_file()}
    from rhythmsmith_core import addons, stems as core_stems
    model = core_stems.MODEL_6S if guitar else core_stems.MODEL
    out = cache / model
    if not core_stems.have_stems(out, model):
        if not addons.installed("stems"):
            log("stem separation isn't installed: charting from the full mix")
            return {}
        log(f"progress: separating stems ({model})")
        if not addons.separate(audio, out, log, model=model):
            return {}
    return {n: out / f"{n}.ogg" for n in STEM_NAMES if (out / f"{n}.ogg").is_file()}


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
