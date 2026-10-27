# results/

Machine-readable outputs behind the paper's **baseline** comparison. These files
are versioned so that the reference numbers remain checkable without re-running
training.

> **Scope.** This directory publishes baseline rows only. Results for the
> proposed method are withheld until the paper is accepted, and every file here
> has been sanitised to remove absolute paths and internal identifiers.

| file | corresponds to |
|---|---|
| `final_tables.json` / `.txt` | baseline comparison regenerated from the trained checkpoints, plus size- and class-stratified AP |
| `multiseed_results.json` | per-seed AP on the leakage-free clean test split for all nine baselines (seeds 0, 1, 2) |
| `multiseed_report.txt` | three-seed summary, including the "is seed noise smaller than the differences?" check and the pairwise overlap counts |
| `multiseed_extreme.json` / `.txt` | the same three-seed protocol scored on the Extreme-WTVD pool |
| `size_class_rows.json` | AP@50 by object size and by defect class |
| `failure_taxonomy.json` / `.txt` | six-way decomposition (matched, wrong class, class overlap, localization 0.1–0.5 IoU, below operating threshold, invisible) per condition |
| `degraded_baselines.json` | clean vs. degraded evaluation for each baseline |
| `diagnosis.json` | the same detectors scored on the leaky public split vs. the leakage-free re-split — quantifies how much of the published performance is leakage |
| `wtvd_split_stats.json`, `extreme_stats.json` | class and split statistics of the two datasets |
| `SHA256SUMS.txt` | checksums of the two distributed archives |

## Headline baseline numbers

Clean test split (177 images), three-seed mean ± sd, AP@50:

```
YOLO v5   43.04 ± 0.02      DANN       31.73 ± 1.33
YOLO v8   41.56 ± 0.42      Advent     32.66 ± 1.98
RT-DETR   42.14 ± 2.62      YOLO-RAW   41.58 ± 0.61
YOLO 11   41.30 ± 1.67
YOLO 12   41.70 ± 0.92
YOLO 26   42.24 ± 1.57
```

Extreme-WTVD test pool (708 images), three-seed mean ± sd, AP@50:

```
YOLO v5   12.74 ± 2.05      DANN       11.48 ± 1.79
YOLO v8   12.83 ± 1.10      Advent     11.31 ± 2.10
RT-DETR   13.50 ± 1.49      YOLO-RAW   12.07 ± 1.40
YOLO 11   11.34 ± 1.32
YOLO 12   12.90 ± 0.23
YOLO 26   11.05 ± 0.79
```

## Caveat

With three seeds the baseline ordering is indicative, not a ranking: all fifteen
pairwise ±1 sd intervals overlap for AP@50:95, and the seed spread reaches
between half and five-sixths of the total span between the best and worst
detector. See `multiseed_report.txt` for the pairwise overlap counts.
