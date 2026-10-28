"""Train any of the stock detectors from the paper's main table on WTVD.

Usage
    python train_detector.py --model yolo11n --seed 0
    python train_detector.py --model rtdetr-l --seed 0
    python train_detector.py --all            # the six stock detectors, seed 0

The recipe (400 epochs, batch 16, 640 px, SGD lr 0.01 / momentum 0.937 /
weight decay 0.0005, patience 100, AMP) is identical for every detector and
matches the manuscript; per-model overrides (batch size for RT-DETR) live in
config.py.
"""

from __future__ import annotations

import argparse

from config import (MODEL_PAPER_NAME, MODEL_WEIGHTS, MODELS, banner,
                    resolve_weights, train_kwargs)

from ultralytics import YOLO


def train_one(model_key: str, seed: int) -> None:
    if model_key not in MODEL_WEIGHTS:
        raise SystemExit(
            f"unknown model '{model_key}'; choose from {sorted(MODEL_WEIGHTS)}")
    run_name = f"{model_key}_clean_seed{seed}"
    banner(f"TRAIN {MODEL_PAPER_NAME[model_key]} ({model_key}) on WTVD",
           model_key=model_key)
    kwargs = train_kwargs(run_name, model_key=model_key)
    kwargs["seed"] = seed
    YOLO(resolve_weights(MODEL_WEIGHTS[model_key])).train(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11n",
                    help=f"one of {sorted(MODEL_WEIGHTS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true",
                    help="train every stock detector once with --seed")
    args = ap.parse_args()

    if args.all:
        for key, _, _, _ in MODELS:
            train_one(key, args.seed)
    else:
        train_one(args.model, args.seed)


if __name__ == "__main__":
    main()
