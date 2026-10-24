"""Optional add-ons stepgen can download on request, kept out of the
installer:

  stems    vocal separation (Demucs + PyTorch, about 1 GB)
  lyrics   AI verse detection: reads the words off the separated vocals
           (faster-whisper + the Whisper "small" model, about 650 MB)
  gpu      NVIDIA GPU support for both (PyTorch's CUDA build, about 3 GB).
           Only offered when an NVIDIA card is found. A Mac needs no
           download: the regular PyTorch already uses Apple Silicon.
  beats    AI beat tracking with Beat This! (needs stems' PyTorch; about 100 MB)
  notes    AI note detection with basic-pitch, on ONNX Runtime (about 150 MB)

They share one private Python in the user's app-data folder, so the
installed app stays small and nothing touches a Python you may have. Every
place a download comes from is listed in SOURCES, shown before anything is
downloaded and printed again as it happens.

The htdemucs_6s model (guitar and piano stems as well, for band charts) is
fetched on first use, or ahead of time with install_model(MODEL_6S).

    python -m rhythmsmith_core.addons install [stems|lyrics|gpu|beats|notes]   # the app's download buttons
    python -m rhythmsmith_core.addons sources [stems|lyrics|gpu|beats|notes]
    python -m rhythmsmith_core.addons status
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PY_VERSION = "3.12.7"
PY_EMBED = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP = "https://bootstrap.pypa.io/get-pip.py"
PYPI = "https://pypi.org (package files from files.pythonhosted.org)"
PACKAGES = {"stems": ["demucs", "soundfile", "numpy"], "lyrics": ["faster-whisper"],
            "beats": ["beat-this", "soxr", "einops", "rotary-embedding-torch"],
            # basic-pitch without its TensorFlow dependency: it runs its bundled
            # ONNX model with ONNX Runtime instead (--no-deps, then what it needs)
            "notes": ["onnxruntime", "librosa", "mir-eval", "pretty-midi", "resampy<0.4.3", "setuptools<81",
                      "scikit-learn", "scipy"]}
NO_DEPS = {"notes": ["basic-pitch"]}
BEAT_THIS_URL = "https://github.com/CPJKU/beat_this"
BEAT_THIS_CHECKPOINT = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BASIC_PITCH_URL = "https://github.com/spotify/basic-pitch"
DEMUCS_MODEL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
MODEL = "htdemucs"
MODEL_6S = "htdemucs_6s"         # adds guitar and piano stems
DEMUCS_MODELS = {MODEL: DEMUCS_MODEL,
                 MODEL_6S: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"}
WHISPER_REPO = "Systran/faster-whisper-small"
WHISPER_URL = f"https://huggingface.co/{WHISPER_REPO}"
# PyTorch built for CUDA, tried in this order: CUDA 12 first, because
# faster-whisper (CTranslate2) uses the cuBLAS/cuDNN 12 libraries it ships
TORCH_CUDA = "https://download.pytorch.org/whl"
TORCH_CUDA_BUILDS = ("cu128", "cu126", "cu130")
NAMES = {"stems": "Vocal separation", "lyrics": "AI verse detection", "gpu": "NVIDIA GPU support",
         "beats": "AI beat tracking (Beat This!)", "notes": "AI note detection (basic-pitch)"}
SIZES = {"stems": "about 1 GB", "lyrics": "about 650 MB", "gpu": "about 3 GB",
         "beats": "about 100 MB", "notes": "about 150 MB"}
NEEDS_STEMS = {"gpu", "beats"}          # these build on the stems add-on's PyTorch


def sources(kind: str) -> list[tuple[str, str]]:
    """(what, where from) for everything installing `kind` downloads."""
    out = []
    if os.name == "nt" and not _addon_python().is_file():
        out += [(f"Python {PY_VERSION} (embeddable, from python.org)", PY_EMBED),
                ("pip, Python's package installer", GET_PIP)]
    if kind == "stems":
        out += [("Demucs, SoundFile, NumPy and PyTorch (Demucs needs it), from the Python "
                 "Package Index", PYPI),
                ("The htdemucs separation model (Meta)", DEMUCS_MODEL)]
    elif kind == "beats":
        out += [("Beat This! (CPJKU, MIT licence) and what it needs (einops, "
                 "rotary-embedding-torch, soxr), from the Python Package Index", PYPI),
                ("Beat This!'s source", BEAT_THIS_URL),
                ("The Beat This! 'final0' model, from its authors' server", BEAT_THIS_CHECKPOINT)]
    elif kind == "notes":
        out += [("basic-pitch (Spotify, Apache 2.0) with its bundled ONNX model, ONNX Runtime, "
                 "librosa, mir_eval, pretty_midi, resampy, scikit-learn and SciPy, from the "
                 "Python Package Index", PYPI),
                ("basic-pitch's source", BASIC_PITCH_URL)]
    elif kind == "gpu":
        out += [("PyTorch and torchaudio built for NVIDIA CUDA, with the CUDA libraries they "
                 "need, replacing the CPU-only PyTorch (PyTorch's own package server; the first "
                 f"of {', '.join(TORCH_CUDA_BUILDS)} that installs)", f"{TORCH_CUDA}/{TORCH_CUDA_BUILDS[0]}")]
    else:
        out += [("faster-whisper and what it needs (CTranslate2, tokenizers, huggingface-hub, "
                 "PyAV, ONNX Runtime), from the Python Package Index", PYPI),
                ("The Whisper 'small' speech model, converted for faster-whisper (Systran, "
                 "from Hugging Face; the files come from its download servers)", WHISPER_URL)]
    return out


def addon_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "stepgen" / "addons" / "stems"


def whisper_dir() -> Path:
    return addon_dir() / "models" / "faster-whisper-small"


def _addon_python() -> Path:
    d = addon_dir() / "python"
    return d / ("python.exe" if os.name == "nt" else "bin/python3")


def _has_demucs(python: str) -> bool:
    try:
        r = subprocess.run([python, "-c", "import demucs.pretrained, soundfile"],
                           capture_output=True, timeout=120, **_no_window())
        return r.returncode == 0
    except Exception:
        return False


def _no_window() -> dict:
    return {"creationflags": 0x08000000} if os.name == "nt" else {}     # CREATE_NO_WINDOW


def stems_python() -> str | None:
    """A Python that can run stems.py, or None if the add-on isn't installed."""
    if not getattr(sys, "frozen", False):
        try:
            import demucs.pretrained  # noqa: F401
            return sys.executable
        except Exception:
            pass
    p = _addon_python()
    if p.is_file():
        return str(p)
    return None


def lyrics_python() -> str | None:
    """A Python that can run lyrics.py, or None if the add-on isn't installed."""
    if not (whisper_dir() / "model.bin").is_file():
        return None
    if not getattr(sys, "frozen", False):
        try:
            import faster_whisper  # noqa: F401
            return sys.executable
        except Exception:
            pass
    p = _addon_python()
    return str(p) if p.is_file() and (addon_dir() / "lyrics.ok").is_file() else None


def gpu_python() -> str | None:
    """The add-ons' Python if it has the CUDA build of PyTorch, else None."""
    p = stems_python()
    if p is None:
        return None
    if p != sys.executable and not (addon_dir() / "gpu.ok").is_file():
        return None
    try:
        r = subprocess.run([p, "-c", "import torch; print(torch.version.cuda or '')"],
                           capture_output=True, text=True, timeout=120, **_no_window())
        return p if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def extra_python(kind: str) -> str | None:
    """A Python with the "beats" or "notes" add-on, or None if it isn't installed."""
    mod = {"beats": "beat_this", "notes": "basic_pitch"}[kind]
    if not getattr(sys, "frozen", False):
        try:
            __import__(mod)
            return sys.executable
        except Exception:
            pass
    p = _addon_python()
    return str(p) if p.is_file() and (addon_dir() / f"{kind}.ok").is_file() else None


def installed(kind: str = "stems") -> bool:
    if kind == "gpu":
        return gpu_python() is not None
    if kind in ("beats", "notes"):
        return extra_python(kind) is not None
    return (stems_python() if kind == "stems" else lyrics_python()) is not None


def _download(url: str, dest: Path, log) -> None:
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _python(log) -> Path:
    """The add-ons' private Python, set up on first use."""
    d = addon_dir()
    d.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        py_dir = d / "python"
        py = py_dir / "python.exe"
        if not py.is_file():
            z = d / "python-embed.zip"
            _download(PY_EMBED, z, log)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(py_dir)
            z.unlink()
            # the embeddable Python ignores site-packages until "import site" is on
            for pth in py_dir.glob("python*._pth"):
                pth.write_text(pth.read_text().replace("#import site", "import site"))
            gp = d / "get-pip.py"
            _download(GET_PIP, gp, log)
            log("setting up pip")
            subprocess.run([str(py), str(gp), "--no-warn-script-location"], check=True, **_no_window())
            gp.unlink()
    else:
        if getattr(sys, "frozen", False):
            # the packaged Mac app has no Python of its own to make a venv from
            raise RuntimeError("add-ons aren't available in the Mac app yet")
        py = _addon_python()
        if not py.is_file():
            log("creating a private Python environment")
            subprocess.run([sys.executable, "-m", "venv", str(d / "python")], check=True)
    return py


def install(log=print, kind: str = "stems") -> None:
    log(f"{NAMES[kind]}: downloading from")
    for what, url in sources(kind):
        log(f"  {what}: {url}")
    py = _python(log)
    if kind == "gpu":
        _install_cuda_torch(py, log)
        return
    if kind in NEEDS_STEMS and stems_python() is None:
        raise RuntimeError(f"download vocal separation first: {NAMES[kind]} uses its PyTorch")
    for pkg in NO_DEPS.get(kind, []):
        log(f"installing {pkg} from {PYPI}")
        subprocess.run([str(py), "-m", "pip", "install", "--no-warn-script-location", "--no-deps", pkg],
                       check=True, **_no_window())
    log(f"installing {', '.join(PACKAGES[kind])} from {PYPI} (this takes a while)")
    subprocess.run([str(py), "-m", "pip", "install", "--no-warn-script-location", *PACKAGES[kind]],
                   check=True, **_no_window())
    if kind == "stems":
        log(f"fetching the separation model from {DEMUCS_MODEL}")
        install_model(MODEL, log)
        log("vocal separation is ready")
    elif kind in ("beats", "notes"):
        if kind == "beats":
            log(f"fetching the Beat This! model from {BEAT_THIS_CHECKPOINT}")
            subprocess.run([str(py), "-c", "from beat_this.inference import load_checkpoint; "
                            "load_checkpoint('final0')"], check=True, **_no_window())
        (addon_dir() / f"{kind}.ok").write_text("ok")
        log(f"{NAMES[kind]} is ready")
    else:
        log(f"fetching the Whisper model from {WHISPER_URL}")
        subprocess.run([str(py), str(_here() / "lyrics.py"), "fetch", str(whisper_dir())],
                       check=True, **_no_window())
        (addon_dir() / "lyrics.ok").write_text("ok")
        log("AI verse detection is ready")


def install_model(model: str = MODEL, log=print) -> None:
    """Fetch a Demucs model into the add-on's model cache (the stems add-on
    has to be there first). MODEL_6S is the one with guitar and piano."""
    py = stems_python()
    if py is None:
        raise RuntimeError("download vocal separation first")
    log(f"fetching the {model} separation model from {DEMUCS_MODELS[model]}")
    subprocess.run([py, "-c", f"from demucs.pretrained import get_model; get_model({model!r})"],
                   check=True, **_no_window())


def _install_cuda_torch(py: Path, log) -> None:
    """Swap the add-ons' CPU-only PyTorch for the CUDA build. Vocal separation
    has to be there first (it brings the rest of what Demucs needs)."""
    if stems_python() is None:
        raise RuntimeError("download vocal separation first: GPU support speeds it up")
    for build in TORCH_CUDA_BUILDS:
        url = f"{TORCH_CUDA}/{build}"
        log(f"installing PyTorch for CUDA ({build}) from {url} (this takes a while)")
        r = subprocess.run([str(py), "-m", "pip", "install", "--no-warn-script-location", "--upgrade",
                            "--force-reinstall", "torch", "torchaudio", "--index-url", url], **_no_window())
        if r.returncode == 0:
            break
        log(f"no {build} build for this Python; trying the next one")
    else:
        raise RuntimeError("no CUDA build of PyTorch installed")
    r = subprocess.run([str(py), "-c", "import torch; print(torch.cuda.get_device_name(0) "
                        "if torch.cuda.is_available() else '')"], capture_output=True, text=True, **_no_window())
    name = r.stdout.strip()
    (addon_dir() / "gpu.ok").write_text("ok")
    log(f"GPU support is ready: {name}" if name else
        "GPU support is installed, but PyTorch can't see the GPU (an old NVIDIA driver?); "
        "stepgen keeps using the CPU until it can")


def _here() -> Path:
    """Where stems.py and lyrics.py are, to run with the add-ons' Python: this
    package's folder (also in a PyInstaller build, which ships them there as
    plain files)."""
    return Path(__file__).resolve().parent


def separate(song: Path, out_dir: Path, log=print, model: str = MODEL) -> bool:
    """Run stems.py with the add-on's Python. False if the add-on isn't there.
    model=MODEL_6S adds guitar and piano stems."""
    py = stems_python()
    if py is None:
        return False
    r = subprocess.run([py, str(_here() / "stems.py"), str(song), str(out_dir), "--model", model],
                       capture_output=True, text=True, **_no_window())
    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        log("stem separation failed: " + (lines[-1] if lines else "no output"))
        return False
    return True


def run_script(kind: str, script: str, args: list, log=print) -> bool:
    """Run one of this package's scripts with the Python that has add-on `kind`.
    False if the add-on isn't there or the script failed."""
    py = extra_python(kind)
    if py is None:
        return False
    r = subprocess.run([py, str(_here() / script), *map(str, args)], capture_output=True, text=True,
                       **_no_window())
    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        log(f"{NAMES[kind]} failed: " + (lines[-1] if lines else "no output"))
        return False
    return True


def transcribe(vocals: Path, out_json: Path, log=print) -> bool:
    """Run lyrics.py on a vocals stem. False if the add-on isn't there or it failed."""
    py = lyrics_python()
    if py is None:
        return False
    r = subprocess.run([py, str(_here() / "lyrics.py"), str(vocals), str(out_json), str(whisper_dir())],
                       capture_output=True, text=True, **_no_window())
    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        log("reading the lyrics failed: " + (lines[-1] if lines else "no output"))
        return False
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    kind = sys.argv[2] if len(sys.argv) > 2 else "stems"
    if cmd == "install":
        install(kind=kind)
    elif cmd == "sources":
        for what, url in sources(kind):
            print(f"{what}: {url}")
    else:
        for k in NAMES:
            print(NAMES[k], "installed" if installed(k) else "not installed")
        print(addon_dir())
