# Extreme-WTVD: UAV Wind-Turbine Defect Detection under Adverse Imaging Conditions

Companion repository of the paper

> **A Vision-Based UAV Inspection System for Wind Turbine Defect Detection under Adverse Imaging Conditions**
> Wei Li, Shenwang Li, Qiuren Su, Yuyang Zhou, Yunwei Zhang, Junlong Deng, Zhenkang Zhang, Li Liu, and Thomas Xinzhang Wu
> *IEEE IAS Publications* — under peer review

[![Status](https://img.shields.io/badge/Code-Released%20upon%20acceptance-lightgrey)](#release-policy)
[![Dataset](https://img.shields.io/badge/Dataset-Baidu%20Netdisk-orange)](#datasets)
[![Results](https://img.shields.io/badge/Results-Available-green)](#results)

## Release policy

**This paper is under peer review.** The source code of the proposed method and the manuscript
source files are **not** in this repository; both will be released upon acceptance.

| item | status |
|---|---|
| Results reported in the paper | ✅ **in this README** |
| Dataset configuration files (`configs/`) | ✅ in this repository |
| WTVD / Extreme-WTVD datasets | ✅ via the links below |
| Trained model weights | ✅ via the links below |
| **Source code of the proposed method** | ⏳ upon acceptance |
| **Manuscript source files** | ❌ not in this repository (distributed only through the review system) |

## Datasets

| dataset | splits (train / val / test) | images | download |
|---|---|---|---|
| **WTVD** (clean) | 3577 / 444 / 177 | 4198 | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| **Extreme-WTVD** (fog / motion blur / low light / composite) | 14308 / 1776 / 708 per family | 16792 | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |

7 defect classes: `burning, crack, deformity, dirt, oil, peeling, rusty`.

Unzip the archives so that the layout is:

```text
.
├── datasets/
│   ├── WTVD/            # train/ val/ test/ + data.yaml
│   └── Extreme-WTVD/    # fog/ motion/ low/ composite/ combined/ + data.yaml
└── configs/
```

## How to train

Install [Ultralytics](https://docs.ultralytics.com/) and train any detector with the configuration
files in [`configs/`](configs):

```bash
pip install ultralytics==8.4.160
```

```python
from ultralytics import YOLO

m = YOLO("yolo11n.pt")
m.train(data="configs/wtvd.yaml", epochs=400, batch=16, imgsz=640,
        optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=0.0005,
        patience=100, amp=True, seed=0)
```

All baselines in the tables below use this identical recipe.

## How to validate

Evaluate a trained checkpoint on the clean split or on the Extreme-WTVD pool:

```bash
yolo detect val model=runs/train/weights/best.pt data=configs/wtvd.yaml
yolo detect val model=runs/train/weights/best.pt data=configs/extreme_wtvd.yaml
```

## Trained models

| model | weights |
|---|---|
| YOLOv5n / v8n / 11n / 12n / 26n, RT-DETR (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| Proposed model | released upon acceptance |

## Results

Results reported in the paper. Clean test split: 177 images; Extreme-WTVD test pool: 708 images.
Values are mean ± standard deviation over three training runs (seeds 0/1/2); rows marked † are a
single run.

**Table 1 — Clean test split (AP in %)**

| method | backbone | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|
| YOLO v5 | YOLO v5n | 47.87 ± 1.81 | 47.88 ± 2.60 | 43.04 ± 0.02 | 17.00 ± 0.52 |
| YOLO v8 | YOLO v8n | 48.38 ± 3.61 | 42.55 ± 3.05 | 41.56 ± 0.42 | 16.09 ± 0.61 |
| RT-DETR | RT-DETR-l | 45.25 ± 1.81 | 45.57 ± 3.02 | 42.14 ± 2.62 | 15.95 ± 0.94 |
| YOLO 11 | YOLO 11n | 47.23 ± 6.67 | 43.64 ± 3.07 | 41.30 ± 1.67 | 16.39 ± 0.96 |
| YOLO 12 | YOLO 12n | 50.67 ± 5.84 | 42.03 ± 3.47 | 41.70 ± 0.92 | 16.75 ± 0.16 |
| YOLO 26 | YOLO 26n | 51.67 ± 2.74 | 42.70 ± 1.85 | 42.24 ± 1.57 | 17.07 ± 0.80 |
| DANN † | YOLO 11n | 45.17 | 41.50 | 31.73 | 11.68 |
| Advent † | YOLO 11n | 44.21 | 41.01 | 32.66 | 11.45 |
| YOLO-RAW † | YOLO 11n | 52.11 | 44.45 | 41.58 | 14.81 |

**Table 2 — Extreme-WTVD test pool (AP in %)**

| method | backbone | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|
| YOLO v5 | YOLO v5n | 26.01 ± 3.12 | 17.64 ± 0.79 | 12.74 ± 2.05 | 4.48 ± 0.89 |
| YOLO v8 | YOLO v8n | 30.43 ± 6.33 | 15.78 ± 1.18 | 12.83 ± 1.10 | 4.64 ± 0.46 |
| RT-DETR | RT-DETR-l | 28.87 ± 3.99 | 19.07 ± 1.31 | 13.50 ± 1.49 | 4.88 ± 0.33 |
| YOLO 11 | YOLO 11n | 27.71 ± 3.37 | 15.04 ± 1.15 | 11.34 ± 1.32 | 4.25 ± 0.55 |
| YOLO 12 | YOLO 12n | 26.98 ± 1.72 | 16.96 ± 1.12 | 12.90 ± 0.23 | 4.78 ± 0.10 |
| YOLO 26 | YOLO 26n | 27.14 ± 1.65 | 15.58 ± 1.08 | 11.05 ± 0.79 | 4.10 ± 0.34 |
| DANN † | YOLO 11n | 28.73 | 16.81 | 11.48 | 4.18 |
| Advent † | YOLO 11n | 28.31 | 15.42 | 11.31 | 4.13 |
| YOLO-RAW † | YOLO 11n | 25.23 | 16.63 | 12.07 | 4.11 |

Every baseline loses most of its accuracy on the synthetic pool — this is the domain gap the
benchmark is designed to expose.

## Repository layout

```text
.
├── configs/
│   ├── wtvd.yaml                 # WTVD split descriptor
│   ├── extreme_wtvd.yaml         # four degradation families + combined pool
│   └── degradation_params.json   # degradation operator parameters
├── CITATION.cff
├── LICENSE                       # MIT (repository material)
├── datasets.LICENSE              # CC BY 4.0 (datasets)
└── README.md
```

## Citation

See [`CITATION.cff`](CITATION.cff). Until the paper is published, please cite it as a manuscript
under review.

## License

* Repository material: MIT — see [`LICENSE`](LICENSE).
* WTVD / Extreme-WTVD: CC BY 4.0 — see [`datasets.LICENSE`](datasets.LICENSE).

## Contact

Shenwang Li (`lishenwang@gxu.edu.cn`), or open an issue.
