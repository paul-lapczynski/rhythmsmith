"""Hardware acceleration: which processor the heavy optional parts run on.

Only the parts that run neural networks can use a GPU: vocal separation
(Demucs, PyTorch), AI verse detection (Whisper, CTranslate2) and AI assist
(ONNX Runtime). Making the charts themselves is plain NumPy on the CPU.

The choice comes from the STEPGEN_DEVICE environment variable ("auto", the
default, or "cpu"), which the app sets per run and `stepgen.py --device`
sets for the command line; the add-ons' own Python inherits it. With
"auto", each part looks for, in order:

    NVIDIA GPU (CUDA)        PyTorch, Whisper, AI assist (when their GPU builds are there)
    Apple Silicon GPU (MPS)  PyTorch; AI assist through Core ML
    DirectML (Windows)       AI assist, when onnxruntime-directml is installed

and falls back to the CPU if none is there or the GPU fails (out of memory,
a missing driver library): a GPU only ever makes things faster, never
breaks a run.

    python accel.py          # what this computer has, as stepgen sees it
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ENV = "STEPGEN_DEVICE"
CHOICES = ("auto", "cpu")


def wanted() -> str:
    v = os.environ.get(ENV, "auto").strip().lower()
    return v if v in CHOICES else "auto"


def torch_device() -> str:
    """'cuda', 'mps' or 'cpu' for PyTorch (vocal separation)."""
    if wanted() == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # the few operations MPS lacks run on the CPU instead of failing
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
    except Exception:
        pass
    return "cpu"


def whisper_device() -> tuple[str, str]:
    """(device, compute type) for faster-whisper. CTranslate2 has CUDA but
    no Apple GPU support, so a Mac reads lyrics on the CPU."""
    if wanted() == "cpu":
        return "cpu", "int8"
    _cuda_dlls()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _cuda_dlls() -> None:
    """On Windows, CTranslate2 needs cuBLAS and cuDNN; the CUDA build of
    PyTorch (the GPU download) ships both, so let Windows find them there."""
    if os.name != "nt":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(lib):
                os.add_dll_directory(lib)
                os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


ONNX_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")


def onnx_providers() -> list[str]:
    """ONNX Runtime providers for AI assist, best first, always ending on the CPU."""
    if wanted() == "cpu":
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        have = set(ort.get_available_providers())
    except Exception:
        have = set()
    return [p for p in ONNX_ORDER if p in have] + ["CPUExecutionProvider"]


def nvidia_gpu() -> str | None:
    """The NVIDIA card's name if the driver is installed (nvidia-smi), else None.
    Doesn't need PyTorch, so the app can ask before anything is downloaded."""
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = p if os.path.isfile(p) else None
    if exe is None:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10,
                           **({"creationflags": 0x08000000} if os.name == "nt" else {}))
        name = r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
        return name or None
    except Exception:
        return None


def apple_silicon() -> bool:
    return sys.platform == "darwin" and os.uname().machine == "arm64"


def gpu_name() -> str | None:
    """A short name for the GPU stepgen could use, or None."""
    n = nvidia_gpu()
    if n:
        return n
    if apple_silicon():
        return "Apple Silicon GPU"
    return None


def report() -> list[str]:
    """One line per part: where it runs on this computer right now."""
    out = [f"setting: {wanted()}", f"GPU found: {gpu_name() or 'none'}"]
    out.append(f"vocal separation (PyTorch): {torch_device()}")
    dev, ct = whisper_device()
    out.append(f"AI verse detection (Whisper): {dev} ({ct})")
    out.append(f"AI assist (ONNX Runtime): {onnx_providers()[0]}")
    return out


if __name__ == "__main__":
    print("\n".join(report()))
