# run from a checkout without installing anything: bandsmith/ and core/ on the path
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
for p in (root / "bandsmith", root / "core", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SCAN_CHART = root / "bandsmith" / "tools" / "scan-chart"


def scan_chart_cmd():
    """node check.mjs, if Node and scan-chart (npm ci in tools/scan-chart) are there."""
    node = shutil.which("node")
    if node is None or not (SCAN_CHART / "node_modules" / "scan-chart").is_dir():
        return None
    return [node, str(SCAN_CHART / "check.mjs")]
