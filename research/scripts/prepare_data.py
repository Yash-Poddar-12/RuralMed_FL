"""Foreground, unattended Phase 1 data entry point; run from any directory."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.src.data_prep.prepare import main

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
