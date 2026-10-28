# Baseline training and evaluation code

Reference implementation of the **comparison methods** reported in the paper.
The proposed method itself is *not* part of this repository and will be
released upon acceptance.

| family | script(s) | what they implement |
|---|---|---|
| Stock detectors (YOLO v5/v8/11/12/26n, RT-DETR) | `train_detector.py` | identical recipe for all six (400 epochs, batch 16, 640 px, SGD lr 0.01 / momentum 0.937 / weight decay 0.0005, patience 100, AMP) |
| YOLO-RAW | `train_yoloraw.py` + `yoloraw_model.py`, `yoloraw_modules.py`, `yolorawn.yaml` | re-implementation of the adverse-weather model on a YOLOv5n backbone |
| DANN / Advent (domain adaptation) | `da/train_da.py`, `da/eval_da.py` | gradient-reversal (Ganin et al.) and virtual-adversarial (Vu et al.) adaptation of a converged clean-trained YOLO 11n on the Extreme-WTVD families |

## Setup

```bash
pip install ultralytics==8.4.160   # plus torch with CUDA
```

Unzip the datasets so that the layout is `<repo>/datasets/WTVD/...` and
`<repo>/datasets/Extreme-WTVD/...` (see the top-level README). If the data
lives elsewhere, point the scripts at it with `EWTVD_ROOT=/path/to/repo`.

## 1. Stock detectors

```bash
cd baselines
python train_detector.py --model yolo11n --seed 0     # one detector
python train_detector.py --all --seed 0               # all six
```

Run with `--seed 1` / `--seed 2` to reproduce the mean ± standard deviation
in the paper's tables.

## 2. YOLO-RAW

```bash
cd baselines
python train_yoloraw.py                # default scale n
yolo detect val model=runs/yolorawn_clean_seed0/weights/best.pt \
                data=../configs/wtvd.yaml
```

## 3. DANN / Advent

Both methods follow the canonical UDA protocol: they start from the
converged clean-trained YOLO 11n (`runs/yolo11n_clean_seed0/weights/best.pt`
from step 1) and fine-tune for 30 epochs with lr 0.001 and the domain weight
ramped 0 → 0.5 (Ganin schedule); the source domain is the clean train split,
the target the four Extreme-WTVD families.

```bash
cd baselines
python da/train_da.py --method dann   --epochs 30 --lr 0.001 --lambda-dom 0.5 \
    --targets fog,motion,low,composite \
    --init-from ../runs/yolo11n_clean_seed0/weights/best.pt --seed 0
python da/train_da.py --method advent --epochs 30 --lr 0.001 --lambda-dom 0.5 \
    --targets fog,motion,low,composite \
    --init-from ../runs/yolo11n_clean_seed0/weights/best.pt --seed 0

python da/eval_da.py --ckpt runs/da/dann_seed0/weights/best.pt --tag dann
```

Note: `train_da.py` loads its source checkpoint with `torch.load`, so pass a
real `.pt` file (e.g. the checkpoint downloaded from the trained-models link,
or produced by step 1) rather than letting Ultralytics auto-download.

## Notes

* Outputs land in `<repo>/runs/` (git-ignored).
* `--verify` in `da/train_da.py` runs a short zero-adaptation-weight pass and
  checks the detection loss against a plain Ultralytics run — a cheap guard
  that the custom loop has not silently diverged from the reference recipe.
