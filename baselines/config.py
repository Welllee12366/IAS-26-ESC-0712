# -*- coding: utf-8 -*-
"""Common configuration for the WTVD baseline training runs.

Hyper-parameters follow the manuscript (Sec. IV-C):

    "All models were trained in a PyTorch 2.0 environment on an NVIDIA Tesla
     V100 (24 GB) for 400 epochs. Stochastic Gradient Descent (SGD) was used
     as the optimizer for all models, with an initial learning rate of 0.01,
     momentum 0.937 and weight decay 0.0005."

Two switches are exposed rather than hidden:
  * DEVICE / BATCH  -- results in the paper were produced on a V100; any
    24 GB GPU gives the same schedule (RT-DETR is trained with a halved
    batch, see BATCH_OVERRIDES).
  * SEEDS           -- seed 0 by default; set (0, 1, 2) for the mean +/- sd
    the tables report.

Model zoo (manuscript main table):
  YOLO v5  -> yolov5nu.pt    (Ultralytics'20)
  YOLO v8  -> yolov8n.pt     (Ultralytics'23)
  YOLO 11  -> yolo11n.pt     (Ultralytics'24)
  YOLO 12  -> yolo12n.pt     (Ultralytics'25)
  YOLO 26  -> yolo26n.pt     (Ultralytics'26)
  RT-DETR  -> rtdetr-l.pt    (CVPR'24, Large)
"""

from pathlib import Path

import sys as _sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from _paths import ROOT  # repository root; override with EWTVD_ROOT

WORKSPACE = ROOT

# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
DATA_YAML = ROOT / "configs" / "wtvd.yaml"
RUNS = ROOT / "runs"          # all training output lands here

# ----------------------------------------------------------------------------
# training hyper-parameters  (manuscript Sec. IV-C)
# ----------------------------------------------------------------------------
EPOCHS = 400
IMGSZ = 640
OPTIMIZER = "SGD"
LR0 = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
PATIENCE = 100              # early-stopping patience
SEED = 0                    # single-seed default
SEEDS = (0,)                # e.g. (0, 1, 2) for the multi-seed study
BATCH = 16
# RT-DETR-L is ~13x larger than the nano YOLOs: at batch 16 it needs ~12.6 GB
# and can overflow when several runs share a GPU. Train it alone with a halved
# batch and fewer dataloader workers.
BATCH_OVERRIDES = {
    "rtdetr-l": 8,
}
WORKERS_OVERRIDES = {
    "rtdetr-l": 4,
}
WORKERS = 8
AMP = True                  # mixed precision (paper: FP16/FP32)
CLOSE_MOSAIC = 10
COS_LR = False

# ----------------------------------------------------------------------------
# the model zoo used in the main table
# ----------------------------------------------------------------------------
MODELS = [
    # key            weights file      paper name    detector family
    ("yolov5n",      "yolov5nu.pt",    "YOLO v5",    "cnn"),
    ("yolov8n",      "yolov8n.pt",     "YOLO v8",    "cnn"),
    ("yolo11n",      "yolo11n.pt",     "YOLO 11",    "cnn"),
    ("yolo12n",      "yolo12n.pt",     "YOLO 12",    "cnn"),
    ("yolo26n",      "yolo26n.pt",     "YOLO 26",    "cnn"),
    ("rtdetr-l",     "rtdetr-l.pt",    "RT-DETR",    "transformer"),
]

# Convenience lookup helpers -------------------------------------------------
MODEL_WEIGHTS = {k: w for k, w, _, _ in MODELS}
MODEL_PAPER_NAME = {k: p for k, _, p, _ in MODELS}
MODEL_FAMILY = {k: f for k, _, _, f in MODELS}


def resolve_weights(weights_file):
    """Prefer a locally downloaded checkpoint, else fall back to the bare name
    (which makes Ultralytics attempt its own download)."""
    local = weights_dir() / weights_file
    if local.is_file():
        return str(local)
    return weights_file


def train_kwargs(run_name, model_key=None):
    """Build the Ultralytics .train(**kwargs) dict shared by every variant."""
    batch = BATCH_OVERRIDES.get(model_key, BATCH) if model_key else BATCH
    workers = WORKERS_OVERRIDES.get(model_key, WORKERS) if model_key else WORKERS
    return dict(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch,
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        seed=SEED,
        workers=workers,
        amp=AMP,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        project=str(RUNS),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        val=True,
        verbose=True,
        plots=True,
    )


def banner(title, model_key=None):
    line = "=" * 78
    eff_batch = BATCH_OVERRIDES.get(model_key, BATCH) if model_key else BATCH
    eff_workers = WORKERS_OVERRIDES.get(model_key, WORKERS) if model_key else WORKERS
    print(line)
    print(title)
    print(line)
    print("data        :", DATA_YAML)
    print("epochs      :", EPOCHS)
    print("imgsz       :", IMGSZ)
    print("optimizer   :", OPTIMIZER, " lr0:", LR0,
          " momentum:", MOMENTUM, " weight_decay:", WEIGHT_DECAY)
    print("batch       :", eff_batch, " workers:", eff_workers, " amp:", AMP)
    print("seeds       :", SEEDS)
    print("output      :", RUNS)
    print(line)
