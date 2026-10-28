# -*- coding: utf-8 -*-
"""
Train the YOLO-RAW reproduction on WTVD.

Paper reference : Table IV, row "YOLO-RAW / IEEE T-ITS'25"
                  (Munir et al., "YOLO-RAW: Advancing UAV Detection With
                   Robustness to Adverse Weather Conditions")
Authors' code   : not released -> this is a re-implementation of the published
                  architecture (see yoloraw_model.py / yoloraw_modules.py).

Protocol
    The model is trained with exactly the same recipe as every other row of
    Table IV (config.py: 400 epochs, 640 px, SGD lr0 0.01 / momentum 0.937 /
    weight decay 0.0005, batch 16, AMP, patience 100) so that the comparison
    isolates the architecture. Two deliberate deviations from the *original*
    paper are recorded as such in the manuscript prose:
      * unified 400-epoch schedule (the authors used 200 epochs, step decay),
      * the Ultralytics DFL / anchor-free head in place of the anchor-based
        YOLOv5 head.

Pretrained
    Where a COCO checkpoint of the matching YOLOv5 scale is available it is
    partially transferred into the shared backbone layers (`model.load` reports
    how many items matched); the YOLO-RAW-specific layers are trained from
    scratch. This mirrors the authors' use of COCO pre-trained weights.

Run
    python train_yoloraw.py                 # paper scale (m), all SEEDS
    python train_yoloraw.py --scale n       # capacity-matched variant
    python train_yoloraw.py --epochs 2 --imgsz 320 --batch 8   # smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import RUNS, SEEDS, banner, train_kwargs  # noqa: E402
from yoloraw_model import dataset_nc, write_cfg  # noqa: E402
from yoloraw_modules import register  # noqa: E402

register()

from ultralytics import YOLO  # noqa: E402

# The authors' YOLO-RAW is built on YOLOv5m. We train the capacity-matched
# **nano** scale instead (5.2 M / 7.9 GFLOPs) so that the Table IV row is
# compute-comparable with the other rows (YOLOv5nu is 7.7 GFLOPs), and record
# that substitution explicitly in the manuscript prose.
PAPER_SCALE = "n"       # the paper's own scale, for reference: "m"


def coco_checkpoint(scale: str) -> Path | None:
    """Locally downloaded Ultralytics YOLOv5 COCO checkpoint for this scale."""
    cand = HERE / "weights" / f"yolov5{scale}u.pt"
    if cand.is_file():
        return cand
    cand = HERE / "weights" / f"yolov5{scale}.pt"
    return cand if cand.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=PAPER_SCALE, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None,
                    help="fraction of the training set (smoke tests only)")
    ap.add_argument("--name", default=None, help="override the run name")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    scale = args.scale
    nc = dataset_nc()
    cfg = write_cfg(scale, nc)

    banner(f"TRAIN YOLO-RAW (re-implementation, YOLOv5{scale} backbone) on WTVD")
    print("model cfg   :", cfg)
    print("nc          :", nc)

    for seed in args.seeds:
        run_name = args.name or f"yoloraw{scale}_clean_seed{seed}"
        print(f"\n--- seed {seed} -> {run_name} ---")

        kw = train_kwargs(run_name, model_key="yoloraw")
        kw["seed"] = seed
        if args.epochs is not None:
            kw["epochs"] = args.epochs
        if args.imgsz is not None:
            kw["imgsz"] = args.imgsz
        if args.batch is not None:
            kw["batch"] = args.batch
        if args.workers is not None:
            kw["workers"] = args.workers
        if args.frac is not None:
            kw["fraction"] = args.frac
        if args.device is not None:
            kw["device"] = args.device
        if args.epochs is not None and args.epochs < kw["patience"]:
            kw["patience"] = args.epochs  # smoke tests must not early-stop weirdly

        model = YOLO(str(cfg))
        if not args.no_pretrained:
            ckpt = coco_checkpoint(scale)
            if ckpt is not None:
                print(f"transferring COCO weights from {ckpt.name} ...")
                try:
                    model.load(str(ckpt))
                except Exception as exc:  # noqa: BLE001
                    print(f"  partial COCO transfer failed ({exc}); training from scratch")
            else:
                print(f"no COCO checkpoint for scale {scale} in weights/; "
                      f"training from scratch")
        model.train(**kw)
        print(f"done: {run_name}")


if __name__ == "__main__":
    main()
