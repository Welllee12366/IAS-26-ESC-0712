# Extreme-WTVD: UAV Wind-Turbine Defect Detection under Adverse Imaging Conditions

Companion repository of the paper

> **A Vision-Based UAV Inspection System for Wind Turbine Defect Detection under Adverse Imaging Conditions**
>   
> Wei Li, Shenwang Li, Qiuren Su, Yuyang Zhou, Yunwei Zhang, Junlong Deng, Zhenkang Zhang, Li Liu, and Thomas Xinzhang Wu
>   
> *IEEE IAS Publications* — under peer review

![Status](https://img.shields.io/badge/Code-Released%20upon%20acceptance-lightgrey)

![Dataset](https://img.shields.io/badge/Dataset-Baidu%20Netdisk-orange)

![Results](https://img.shields.io/badge/Results-Available-green)

## Release policy

**This paper is under peer review.** The source code of the proposed method and the manuscript
source files are **not** in this repository; both will be released upon acceptance.

| item                                     | status                                                                |
| ---------------------------------------- | --------------------------------------------------------------------- |
| Results reported in the paper            | ✅ **in this README**                                                  |
| **Baseline training / evaluation code**  | ✅ **in `baselines/`**                                                 |
| Dataset configuration files (`configs/`) | ✅ in this repository                                                  |
| WTVD / Extreme-WTVD datasets             | ⏳ upon acceptance                                                     |
| Trained model weights                    | ✅ via the links below                                                 |
| **Source code of the proposed method**   | ⏳ upon acceptance                                                     |

## Datasets

| dataset                                                      | splits (train / val / test)   | images | download              |
| ------------------------------------------------------------ | ----------------------------- | ------ | ---------------------- |
| **WTVD** (clean)                                             | 3577 / 444 / 177              | 4198   | ⏳ upon acceptance     |
| **Extreme-WTVD** (fog / motion blur / low light / composite) | 14308 / 1776 / 708 per family | 16792  | ⏳ upon acceptance     |

7 defect classes: `burning, crack, deformity, dirt, oil, peeling, rusty`.

Unzip the archives so that the layout is:

```text
.
├── datasets/
│   ├── WTVD/                        # clean, leakage-free
│   │   ├── train/
│   │   │   ├── images/              # 3577
│   │   │   └── labels/
│   │   ├── val/                     # images/ labels/ — 444
│   │   ├── test/                    # images/ labels/ — 177
│   │   └── wtvd.yaml
│   └── Extreme-WTVD/                # synthetic degradation benchmark
│       ├── fog/
│       │   ├── train/
│       │   │   ├── images/          # 3577
│       │   │   └── labels/
│       │   └── val/                 # images/ labels/ — 444
│       ├── motion/                  # train/ val/ test/ — same structure
│       ├── low/                     # idem
│       ├── composite/               # idem
│       ├── combined/                # all four families pooled — 14308 / 1776 / 708
│       └── extreme_wtvd.yaml
└── configs/
```

## How to train

The comparison-method code ships in [`baselines/`](baselines) — see
[`baselines/README.md`](baselines/README.md) for the full usage. In short:

```bash
pip install ultralytics==8.4.160
cd baselines
python train_detector.py --model yolo11n --seed 0      # any of the six stock detectors
```

YOLO-RAW is trained with `train_yoloraw.py`; DANN / Advent with
`da/train_da.py` (starting from the converged clean-trained YOLO 11n).
All use the same recipe, so the tables below are reproducible.

## How to validate

Evaluate a trained checkpoint on the clean split or on the Extreme-WTVD pool:

```bash
yolo detect val model=runs/train/weights/best.pt data=configs/wtvd.yaml
yolo detect val model=runs/train/weights/best.pt data=configs/extreme_wtvd.yaml
```

## Trained models

| model | weights |
|---|---|
| YOLO v5n (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| YOLO v8n (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| YOLO 11n (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| YOLO 12n (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| YOLO 26n (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| RT-DETR (clean-trained) | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| DANN / Advent / YOLO-RAW | [Baidu Netdisk](https://pan.baidu.com/s/xxxxxxxxxxxxxx) (code: `xxxx`) |
| Proposed model (Ours) | released upon acceptance |

## Results

Results reported in the paper (Table "Performance Results on WTVD and Extreme-WTVD Datasets").
All models were trained three times under an identical recipe, so both tables report the
mean ± sample standard deviation over seeds 0, 1 and 2.

**Table 1 — WTVD, clean test split (%, 177 images)**

| method | backbone | Params (M) | GFLOPs | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|---|---|
| YOLO v5 | YOLO v5n | 2.51 | 7.18 | 47.87 ± 1.81 | 47.88 ± 2.60 | 43.04 ± 0.02 | 17.00 ± 0.52 |
| YOLO v8 | YOLO v8n | 3.01 | 8.20 | 48.38 ± 3.61 | 42.55 ± 3.05 | 41.56 ± 0.42 | 16.09 ± 0.61 |
| RT-DETR | RT-DETR-l | 32.82 | 109.92 | 45.25 ± 1.81 | 45.57 ± 3.02 | 42.14 ± 2.62 | 15.95 ± 0.94 |
| YOLO 11 | YOLO 11n | 2.59 | 6.51 | 47.23 ± 6.67 | 43.64 ± 3.07 | 41.30 ± 1.67 | 16.39 ± 0.96 |
| YOLO 12 | YOLO 12n | 2.57 | 7.47 | 50.67 ± 5.84 | 42.03 ± 3.47 | 41.70 ± 0.92 | 16.75 ± 0.16 |
| YOLO 26 | YOLO 26n | 2.51 | 5.91 | 51.67 ± 2.74 | 42.70 ± 1.85 | 42.24 ± 1.57 | 17.07 ± 0.80 |
| DANN | YOLO 11n | 2.59 | 6.51 | 45.17 ± 4.15 | 41.50 ± 3.26 | 31.73 ± 1.33 | 11.68 ± 1.21 |
| Advent | YOLO 11n | 2.59 | 6.51 | 44.21 ± 5.82 | 41.01 ± 4.16 | 32.66 ± 1.98 | 11.45 ± 1.69 |
| YOLO-RAW | YOLO 11n | 5.22 | 7.89 | 52.11 ± 3.79 | 44.45 ± 3.26 | 41.58 ± 0.61 | 14.81 ± 0.85 |
| **Ours** | YOLO 11n | 3.11 | 6.81 | 61.65 ± 2.31 | 55.04 ± 2.93 | 50.11 ± 0.61 | 23.77 ± 0.41 |

**Table 2 — Extreme-WTVD test pool (%, 708 images)**

| method | backbone | Precision | Recall | AP@50 | AP@50:95 |
|---|---|---|---|---|---|
| YOLO v5 | YOLO v5n | 26.01 ± 3.12 | 17.64 ± 0.79 | 12.74 ± 2.05 | 4.48 ± 0.89 |
| YOLO v8 | YOLO v8n | 30.43 ± 6.33 | 15.78 ± 1.18 | 12.83 ± 1.10 | 4.64 ± 0.46 |
| RT-DETR | RT-DETR-l | 28.87 ± 3.99 | 19.07 ± 1.31 | 13.50 ± 1.49 | 4.88 ± 0.33 |
| YOLO 11 | YOLO 11n | 27.71 ± 3.37 | 15.04 ± 1.15 | 11.34 ± 1.32 | 4.25 ± 0.55 |
| YOLO 12 | YOLO 12n | 26.98 ± 1.72 | 16.96 ± 1.12 | 12.90 ± 0.23 | 4.78 ± 0.10 |
| YOLO 26 | YOLO 26n | 27.14 ± 1.65 | 15.58 ± 1.08 | 11.05 ± 0.79 | 4.10 ± 0.34 |
| DANN | YOLO 11n | 28.73 ± 5.78 | 16.81 ± 1.22 | 11.48 ± 1.79 | 4.18 ± 0.46 |
| Advent | YOLO 11n | 28.31 ± 5.41 | 15.42 ± 1.70 | 11.31 ± 2.10 | 4.13 ± 0.88 |
| YOLO-RAW | YOLO 11n | 25.23 ± 3.13 | 16.63 ± 1.13 | 12.07 ± 1.40 | 4.11 ± 0.61 |
| **Ours** | YOLO 11n | 42.20 ± 2.79 | 37.56 ± 0.83 | 32.99 ± 0.95 | 12.18 ± 0.48 |

Every baseline loses most of its accuracy on the synthetic pool — this is the domain gap the
benchmark is designed to expose.

## Repository layout

```text
.
├── baselines/                    # training / evaluation code for the comparison methods
│   ├── train_detector.py         # stock YOLO v5/v8/11/12/26n + RT-DETR (paper recipe)
│   ├── config.py, _paths.py      # shared recipe and path resolution
│   ├── train_yoloraw.py, yoloraw_model.py, yoloraw_modules.py, yolorawn.yaml
│   ├── da/
│   │   ├── train_da.py           # DANN / Advent domain adaptation
│   │   └── eval_da.py            # DA checkpoint evaluation
│   └── README.md                 # usage for all of the above
├── dataset_tools/                # benchmark fidelity / integrity measurements
│   ├── measure_fid_kid.py        # FID + KID, clean vs. each degraded family
│   ├── defect_feature_distance.py# paired defect-box feature distance
│   ├── verify_fog_box_ssim.py    # in-box SSIM under fog
│   └── README.md
├── configs/
│   ├── wtvd.yaml                 # WTVD split descriptor
│   ├── extreme_wtvd.yaml         # four degradation families + combined pool
│   └── degradation_params.json   # degradation operator parameters
├── CITATION.cff
├── LICENSE                       # MIT (repository material)
├── datasets.LICENSE              # CC BY 4.0 (datasets)
└── README.md
```

The code of the **proposed method** is not part of this repository; see the release policy.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the paper is published, please cite it as a manuscript
  
under review.

## License

- Repository material: MIT — see [`LICENSE`](LICENSE).
- WTVD / Extreme-WTVD: CC BY 4.0 — see [`datasets.LICENSE`](datasets.LICENSE).

## Contact

Shenwang Li (`lishenwang@gxu.edu.cn`), or open an issue.
