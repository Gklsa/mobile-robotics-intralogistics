#!/usr/bin/env python3
"""Run optional newly written teaching labs, independent of the course Challenge.

Examples:
  python tools/run_labs.py --lab mapping --noise high
  python tools/run_labs.py --lab mapping --left-scale 1.002 --right-scale 0.999
  python tools/run_labs.py --lab bug2 --output outputs/bug2.json
  python tools/run_labs.py --lab line-following --output outputs/line.json

These synthetic scenes are not the original course track. Mapping integrates
encoder odometry and ranges, without observation-based pose correction, loop
closure, or a claim to a complete SLAM implementation. No web runtime is used.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intralogistics_sim.robot import LOW_NOISE, HIGH_NOISE
from intralogistics_sim.scenarios import simulate_mapping, simulate_bug2, simulate_line_following


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and zero or greater")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab", required=True, choices=("mapping", "bug2", "line-following"))
    parser.add_argument("--output", type=Path, help="JSON file or directory; defaults to outputs/labs/<lab>.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", choices=("low", "high"), help="encoder preset for mapping / bug2; default low")
    parser.add_argument("--left-scale", type=positive_float, help="left wheel distance calibration scale")
    parser.add_argument("--right-scale", type=positive_float, help="right wheel distance calibration scale")
    parser.add_argument("--wheelbase-scale", type=positive_float, help="odometry wheelbase calibration scale")
    parser.add_argument("--distance-noise", type=nonnegative_float, help="incremental encoder noise standard deviation, metres")
    parser.add_argument("--encoder-resolution", type=positive_float, help="encoder distance quantisation, metres")
    args = parser.parse_args()

    overrides = {name: value for name, value in (
        ("left_scale", args.left_scale), ("right_scale", args.right_scale),
        ("wheelbase_scale", args.wheelbase_scale), ("distance_noise", args.distance_noise),
        ("resolution", args.encoder_resolution),
    ) if value is not None}
    if args.lab == "line-following" and (args.noise is not None or overrides):
        parser.error("encoder options apply to mapping / bug2; line-following compares the same steering-bias condition")

    parameters = {"seed": args.seed}
    if args.lab == "line-following":
        replay = simulate_line_following(args.seed)
        summary = {run["id"]: run["metrics"] for run in replay["runs"]}
    else:
        noise = args.noise or "low"
        model = replace(HIGH_NOISE if noise == "high" else LOW_NOISE, **overrides)
        parameters.update({"noise_preset": noise, "encoder_model": asdict(model)})
        if args.lab == "mapping":
            replay = simulate_mapping(noise, args.seed, encoder_model=model)
        else:
            replay = simulate_bug2(args.seed, encoder_model=model)
            replay["noise"] = noise
        summary = replay["metrics"]
    replay["lab"] = {"name": args.lab, "parameters": parameters, "original_course_track": False}
    replay.setdefault("notes", []).insert(0,
        "Optional newly written synthetic teaching lab (2026); separate from the original course Challenge map and 2025 hardware work.")
    target = args.output or ROOT / "outputs" / "labs" / f"{args.lab}.json"
    if target.suffix.lower() != ".json":
        target = target / f"{args.lab}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(target.resolve()), "lab": args.lab,
                      "parameters": parameters, "metrics": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
