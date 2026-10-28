# -*- coding: utf-8 -*-
"""
YOLO-RAW (Munir et al., IEEE T-ITS 2025) re-implementation, Ultralytics dialect.

Published architecture (Sec. III, Figs. 3-4):

  Backbone  CSPDarknet53 (YOLOv5) + one extra CBS and one extra C3 block,
            followed by SPP-Extended at the end of the backbone.
  Neck      Bi-FPN-like, extended with a fourth fusion stage Nu6 that merges the
            Nu5 feature maps with Nd6 coming from after SPP-Extended, so that
            the deepest (P6) feature map is fused as well; lateral connections
            bring P4 into Nu4 and P5 into Nu5. Four *parameter-free* SimAM
            attention modules are inserted in the neck.
  Head      Four detection heads at P3/P4/P5/P6 (80x80, 40x40, 20x20, 10x10).

Everything here is one-to-one with those statements. The only deliberate
substitution is the detection head itself: Ultralytics 8.x ships a DFL /
anchor-free `Detect`, whereas the paper's anchor-based YOLOv5 head (and its
Table I anchor set) is not available in this framework. Using the same head as
every other row of Table IV keeps the comparison apples-to-apples.

Because the two custom blocks are channel-preserving and are *not* in
parse_model's `base_modules`, parse_model will not prepend `c1`. Their YAML rows
therefore carry the already width-scaled channel numbers, which this module
computes for the requested scale.
"""
from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# YOLOv5 scale factors (depth, width), as used by ultralytics' yolov5*.yaml
SCALES = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}


def _div(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def dataset_nc(data_yaml: str | Path | None = None) -> int:
    """Read `nc` from the dataset yaml, so the model and the data always agree."""
    import yaml

    if data_yaml is None:
        from config import DATA_YAML

        data_yaml = DATA_YAML
    with open(data_yaml, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    return int(d["nc"])


def build_cfg(scale: str = "m", nc: int | None = None) -> dict:
    """Return the YOLO-RAW model dict for one YOLOv5 scale."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}, got {scale!r}")
    if nc is None:
        nc = dataset_nc()
    depth, width = SCALES[scale]

    # SPP-Extended is channel preserving and outside base_modules -> needs the
    # final (width-scaled) channel count written out explicitly.
    c5 = _div(1024 * width)

    backbone = [
        # --- stock CSPDarknet53 ------------------------------------------------
        [-1, 1, "Conv", [64, 6, 2, 2]],        # 0  P1/2
        [-1, 1, "Conv", [128, 3, 2]],          # 1  P2/4
        [-1, 3, "C3", [128]],                  # 2
        [-1, 1, "Conv", [256, 3, 2]],          # 3  P3/8
        [-1, 6, "C3", [256]],                  # 4  -> P3  (80x80)
        [-1, 1, "Conv", [512, 3, 2]],          # 5  P4/16
        [-1, 9, "C3", [512]],                  # 6  -> P4  (40x40)
        [-1, 1, "Conv", [1024, 3, 2]],         # 7  P5/32
        [-1, 3, "C3", [1024]],                 # 8  -> P5  (20x20)
        # --- YOLO-RAW: one extra CBS (Conv+BN+SiLU) + one extra C3 -----------
        [-1, 1, "Conv", [1024, 3, 2]],         # 9  -> P6  (10x10)
        [-1, 3, "C3", [1024]],                 # 10
        # --- SPP-Extended at the end of the backbone --------------------------
        [-1, 1, "SPPE", [c5, c5, 5]],          # 11 -> Nd6 source
    ]

    head = [
        # --- top-down path: Nu6 ... Nu3 ---------------------------------------
        [-1, 1, "SimAM", []],                  # 12  SimAM #1 on the Nd6 path
        [-1, 1, "Conv", [1024, 1, 1]],         # 13  Nd6 lateral
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 8], 1, "Concat", [1]],           # 15  + P5
        [-1, 3, "C3", [1024, False]],          # 16  Nu5
        [-1, 1, "SimAM", []],                  # 17  SimAM #2  (after up #1)
        [-1, 1, "Conv", [512, 1, 1]],          # 18
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],           # 20  + P4   (lateral P4 -> Nu4)
        [-1, 3, "C3", [512, False]],           # 21  Nu4
        [-1, 1, "SimAM", []],                  # 22  SimAM #3  (after up #2)
        [-1, 1, "Conv", [256, 1, 1]],          # 23
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],           # 25  + P3
        [-1, 3, "C3", [256, False]],           # 26  Nu3  -> P3 head
        [-1, 1, "SimAM", []],                  # 27  SimAM #4  (after up #3)
        # --- bottom-up path ---------------------------------------------------
        [-1, 1, "Conv", [256, 3, 2]],          # 28
        [[-1, 21], 1, "Concat", [1]],          # 29  + Nu4
        [-1, 3, "C3", [512, False]],           # 30  -> P4 head
        [-1, 1, "Conv", [512, 3, 2]],          # 31
        [[-1, 16], 1, "Concat", [1]],          # 32  + Nu5  (lateral P5 -> Nu5)
        [-1, 3, "C3", [1024, False]],          # 33  -> P5 head
        [-1, 1, "Conv", [1024, 3, 2]],         # 34
        [[-1, 11], 1, "Concat", [1]],          # 35  Nu5(down) + Nd6  == the Nu6 stage
        [-1, 3, "C3", [1024, False]],          # 36  Nu6  -> P6 head
        # --- four detection heads --------------------------------------------
        [[26, 30, 33, 36], 1, "Detect", [nc]],  # 37  Detect(P3, P4, P5, P6)
    ]

    return {
        "nc": nc,
        "depth_multiple": depth,
        "width_multiple": width,
        "backbone": backbone,
        "head": head,
    }


def write_cfg(scale: str = "m", nc: int | None = None, out: Path | str | None = None) -> Path:
    """Serialise build_cfg() to a .yaml file and return its path."""
    import yaml

    cfg = build_cfg(scale, nc)
    path = Path(out) if out else HERE / f"yoloraw{scale}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None, width=1000)
    return path


def model(scale: str = "m", nc: int | None = None):
    """Build a YOLO-RAW DetectionModel (registers the custom blocks first)."""
    import sys

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from yoloraw_modules import register

    register()

    from ultralytics import YOLO

    path = write_cfg(scale, nc)
    return YOLO(str(path))
