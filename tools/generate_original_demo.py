#!/usr/bin/env python3
"""Regenerate the original course Challenge simulation data from editable config."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from intralogistics_sim.original_challenge import main


if __name__ == '__main__':
    main()
