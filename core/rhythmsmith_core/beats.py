"""Beats and downbeats with Beat This! (CPJKU, MIT licence), a neural beat
tracker that follows tempo changes. Optional: the "beats" add-on (addons.py)
installs it into the add-ons' own Python, which runs this script:

    python beats.py <audio> <out.json>

Writes {"model": ..., "beats": [seconds], "downbeats": [seconds]}. The first run
downloads the "final0" checkpoint from CHECKPOINT_URL (Beat This!'s own server).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if not __package__:      # run as a script by the add-ons' embedded Python, which leaves
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the script's folder off sys.path

REPO_URL = "https://github.com/CPJKU/beat_this"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
CHECKPOINT = "final0"


def track(audio: Path) -> dict:
    try:
        from . import accel
    except ImportError:     # run as a script (see the top of this file)
        import accel
    import soundfile as sf
    from beat_this.inference import Audio2Beats
    device = accel.torch_device()
    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    try:
        beats, downbeats = Audio2Beats(checkpoint_path=CHECKPOINT, device=device)(y.mean(axis=1), sr)
    except Exception as e:                  # out of GPU memory, a driver problem: the CPU always works
        if device == "cpu":
            raise
        print(f"the GPU ({device}) failed ({str(e).splitlines()[0][:120]}); finding beats on the CPU", flush=True)
        beats, downbeats = Audio2Beats(checkpoint_path=CHECKPOINT, device="cpu")(y.mean(axis=1), sr)
    return {"model": f"beat_this {CHECKPOINT}", "beats": [float(t) for t in beats],
            "downbeats": [float(t) for t in downbeats]}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    Path(sys.argv[2]).write_text(json.dumps(track(Path(sys.argv[1]))))
