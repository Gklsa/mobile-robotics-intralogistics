#!/usr/bin/env python3
"""Regenerate the original-course Challenge replay without PYTHONPATH setup."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intralogistics_sim.original_challenge import main


if __name__ == "__main__":
    main()
