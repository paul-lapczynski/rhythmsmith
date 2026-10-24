"""Optional add-ons stepgen can download on request, kept out of the
installer:

  stems    vocal separation (Demucs + PyTorch, about 1 GB)
  lyrics   AI verse detection: reads the words off the separated vocals
           (faster-whisper + the Whisper "small" model, about 650 MB)
  gpu      NVIDIA GPU support for both (PyTorch's CUDA build, about 3 GB).
           Only offered when an NVIDIA card is found. A Mac needs no
           download: the regular PyTorch already uses Apple Silicon.

They share one private Python in the user's app-data folder, so the
installed app stays small and nothing touches a Python you may have. Every
place a download comes from is listed in SOURCES, shown before anything is
downloaded and printed again as it happens.

The htdemucs_6s model (guitar and piano stems as well, for band charts) is
fetched on first use, or ahead of time with install_model(MODEL_6S).

    python -m rhythmsmith_core.addons install [stems|lyrics|gpu]   # what the app's download buttons run
    python -m rhythmsmith_core.addons sources [stems|lyrics|gpu]
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
PACKAGES = {"stems": ["demucs", "soundfile", "numpy"], "lyrics": ["faster-whisper"]}
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
NAMES = {"stems": "Vocal separation", "lyrics": "AI verse detection", "gpu": "NVIDIA GPU support"}
SIZES = {"stems": "about 1 GB", "lyrics": "about 650 MB", "gpu": "about 3 GB"}


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


def installed(kind: str = "stems") -> bool:
    if kind == "gpu":
        return gpu_python() is not None
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
    log(f"installing {', '.join(PACKAGES[kind])} from {PYPI} (this takes a while)")
    subprocess.run([str(py), "-m", "pip", "install", "--no-warn-script-location", *PACKAGES[kind]],
                   check=True, **_no_window())
    if kind == "stems":
        log(f"fetching the separation model from {DEMUCS_MODEL}")
        install_model(MODEL, log)
        log("vocal separation is ready")
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
