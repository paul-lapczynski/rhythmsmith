"""Split a song into stems (vocals, drums, bass, other) with Demucs, so
stepgen can follow the vocal or lead line instead of everything tonal.

    python stems.py "My Song.ogg"            # writes My Song.stems/ next to it
    python stems.py "My Song.mp3" out_dir

Optional: needs `pip install demucs soundfile` (pulls in PyTorch, about
1 GB); the app can download all of it for you (Options > Vocal separation). The
first run downloads the htdemucs model (~80 MB). Separation takes roughly
one to three minutes per song on a normal CPU, a few seconds on a GPU
(NVIDIA, or Apple Silicon; see accel.py); the result is cached, so each
song is only separated once.

    python stems.py "My Song.ogg" out_dir --model htdemucs_6s

htdemucs_6s (MODEL_6S) also splits out guitar and piano, which band charts
need; the default stays htdemucs, the four stems above.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if not __package__:      # run as a script by the add-ons' embedded Python, which leaves
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the script's folder off sys.path

STEMS = ("vocals", "drums", "bass", "other")
MODEL = "htdemucs"
MODEL_6S = "htdemucs_6s"
STEMS_6S = ("vocals", "drums", "bass", "guitar", "piano", "other")
MODELS = {MODEL: STEMS, MODEL_6S: STEMS_6S}


def stems_for(model: str = MODEL) -> tuple[str, ...]:
    """The stems a model writes."""
    return MODELS[model]


def stems_dir_for(song_folder: Path) -> Path:
    """Where stepgen keeps a song's stems: a hidden folder in its song folder."""
    return song_folder / ".stems"


def have_stems(d: Path, model: str = MODEL) -> bool:
    return all((d / f"{s}.ogg").is_file() for s in stems_for(model))


def demucs_available() -> bool:
    try:
        import demucs.apply  # noqa: F401
        import demucs.pretrained  # noqa: F401
        return True
    except Exception:
        return False


def _write_ogg(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=data.shape[1],
                      format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(data), sr * 10):            # blocks: big Vorbis writes can crash
            f.write(data[i:i + sr * 10])


def separate(audio: Path, out_dir: Path, log=print, model_name: str = MODEL) -> Path:
    """Write vocals/drums/bass/other.ogg into out_dir (skipped if already there);
    guitar.ogg and piano.ogg as well with model_name=MODEL_6S."""
    wanted = stems_for(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if have_stems(out_dir, model_name):
        return out_dir
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    log(f"progress: separating stems ({model_name})")
    model = get_model(model_name)
    model.eval()
    data, sr = sf.read(str(audio), dtype="float32", always_2d=True)   # .ogg/.wav/.flac
    y = data.T
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    if sr != model.samplerate:
        import torchaudio.functional as AF
        y = AF.resample(torch.tensor(y), sr, model.samplerate).numpy()
        sr = model.samplerate
    wav = torch.tensor(y[:2], dtype=torch.float32)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    try:
        from . import accel
    except ImportError:     # run as a script (see the top of this file)
        import accel
    device = accel.torch_device()
    log(f"progress: separating on {device}")
    try:
        out = _apply(apply_model, model, wav, mean, std, device)
    except Exception as e:                  # out of GPU memory, a driver problem: the CPU always works
        if device == "cpu":
            raise
        log(f"the GPU ({device}) failed ({str(e).splitlines()[0][:120]}); separating on the CPU")
        out = _apply(apply_model, model, wav, mean, std, "cpu")
    out = out * std + mean
    for name, src in zip(model.sources, out):
        if name in wanted:
            _write_ogg(out_dir / f"{name}.ogg", src.numpy().T.astype(np.float32), sr)
    return out_dir


def _apply(apply_model, model, wav, mean, std, device: str):
    import torch
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device, split=True,
                          overlap=0.25, progress=False)[0]
    return out.cpu()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    if not demucs_available():
        print("Demucs isn't installed. Run:  pip install demucs")
        return 1
    model = MODEL
    if "--model" in argv:
        i = argv.index("--model")
        model = argv[i + 1] if i + 1 < len(argv) else ""
        argv = argv[:i] + argv[i + 2:]
        if model not in MODELS:
            print(f"--model is one of {', '.join(MODELS)}")
            return 1
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else src.with_suffix(".stems")
    separate(src, out, model_name=model)
    print(f"stems in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
