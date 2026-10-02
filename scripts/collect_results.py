"""Copy the summary result files from experiments/ into results/summaries/.

The analysis scripts read and write their JSON next to the code, so this just
keeps a browsable copy under results/. Run it after re-running any analysis.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments"
DST = ROOT / "results" / "summaries"
DST.mkdir(parents=True, exist_ok=True)

files = sorted(SRC.glob("results_*.json")) + [SRC / "draw_count_calibration.json"]
for f in files:
    shutil.copy2(f, DST / f.name)
    print("copied", f.name)
