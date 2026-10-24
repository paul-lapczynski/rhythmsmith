"""Read the words off a song's separated vocals with Whisper (faster-whisper),
so stepgen can tell a chorus (the same words coming back) from a verse (the
same music with new words). Run by the optional lyrics add-on's own Python:

    python lyrics.py fetch <model folder>                  # download the model
    python lyrics.py <vocals.ogg> <out.json> <model folder>

The model is Whisper "small" converted for faster-whisper, downloaded from
MODEL_URL into the model folder (nowhere else). Writes
{"model", "language", "words": [[start, end, word], ...]} in seconds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if not __package__:      # run as a script by the add-ons' embedded Python, which leaves
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the script's folder off sys.path

MODEL_REPO = "Systran/faster-whisper-small"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"


def fetch(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    print(f"downloading the Whisper model from {MODEL_URL}", flush=True)
    snapshot_download(MODEL_REPO, local_dir=str(model_dir))


def load_16k(path: Path):
    """The file as mono float32 at 16 kHz, what Whisper reads. Decoded with
    SoundFile rather than faster-whisper's own PyAV reader, which breaks on
    newer PyAV releases (open() lost its metadata_errors argument)."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        # resample in the frequency domain: drops what's above 8 kHz cleanly
        n = int(round(len(y) * 16000 / sr))
        Y = np.fft.rfft(y)[: n // 2 + 1]
        y = (np.fft.irfft(Y, n) * (n / len(y))).astype(np.float32)
    return y


def transcribe(vocals: Path, out: Path, model_dir: Path) -> None:
    try:
        from . import accel
    except ImportError:     # run as a script (see the top of this file)
        import accel
    from faster_whisper import WhisperModel
    # an NVIDIA GPU when there is one (accel.py), else the CPU: about a minute for a song
    device, compute = accel.whisper_device()
    audio = load_16k(vocals)
    try:
        segs, info = _read(WhisperModel, model_dir, device, compute, audio)
    except Exception as e:                  # missing cuBLAS/cuDNN, out of memory: use the CPU
        if device == "cpu":
            raise
        print(f"the GPU failed ({str(e).splitlines()[0][:120]}); reading the lyrics on the CPU", flush=True)
        segs, info = _read(WhisperModel, model_dir, "cpu", "int8", audio)
    words = [[round(w.start, 3), round(w.end, 3), w.word.strip()]
             for s in segs for w in (s.words or []) if w.word.strip()]
    out.write_text(json.dumps({"model": MODEL_REPO, "language": info.language, "words": words}))


def _read(WhisperModel, model_dir: Path, device: str, compute: str, audio):
    model = WhisperModel(str(model_dir), device=device, compute_type=compute)
    segs, info = model.transcribe(audio, word_timestamps=True, vad_filter=True)
    return list(segs), info          # transcribe() is lazy: a GPU failure shows up here


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fetch":
        fetch(Path(sys.argv[2]))
    elif len(sys.argv) == 4:
        transcribe(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)
