# run from a checkout without installing anything: bandsmith/ and core/ on the path
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
for p in (root / "bandsmith", root / "core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
