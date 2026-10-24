"""Note events (pitch, start, end) from a stem with basic-pitch (Spotify,
Apache 2.0), on ONNX Runtime with the model that ships inside the package.
Optional: the "notes" add-on (addons.py). Used in-process when basic-pitch is
importable, else run by the add-ons' Python:

    python notes.py <audio> <out.json> [<min Hz> <max Hz>]

Writes {"model": ..., "notes": [[start s, end s, MIDI pitch, amplitude], ...]}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if not __package__:      # run as a script by the add-ons' embedded Python, which leaves
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the script's folder off sys.path


def transcribe(audio: Path, fmin: float | None = None, fmax: float | None = None) -> list[list[float]]:
    import warnings
    warnings.filterwarnings("ignore")
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    _, _, events = predict(str(audio), ICASSP_2022_MODEL_PATH, minimum_frequency=fmin,
                           maximum_frequency=fmax, multiple_pitch_bends=False, melodia_trick=True)
    return sorted([[float(s), float(e), int(p), float(a)] for s, e, p, a, *_ in events])


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) == 5 else (None, None)
    Path(sys.argv[2]).write_text(json.dumps({"model": "basic-pitch icassp_2022",
                                             "notes": transcribe(Path(sys.argv[1]), lo, hi)}))
