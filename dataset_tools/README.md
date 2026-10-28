# Dataset fidelity and integrity tools

Evaluation scripts for the WTVD / Extreme-WTVD benchmarks — the measurements
behind the benchmark-quality checks reported in the paper (fidelity of the
synthesized families, annotation transfer, geometry preservation).

All scripts resolve the repository root automatically (`EWTVD_ROOT` env
variable overrides) and expect the released layout:

```text
<root>/datasets/WTVD/...
<root>/datasets/Extreme-WTVD/...
```

Outputs are written to `<root>/runs/` (git-ignored).

| script | what it measures |
|---|---|
| `measure_fid_kid.py` | FID and KID between clean WTVD and each degraded family (Inception-v3 pool3, ImageNet-pretrained — never trained on WTVD). Three honest views: FID on the training split with a disjoint-half null calibration; FID on the 177-image test split after PCA reduction, each row shown with its own null (finite-sample bias makes plain test-split FID untrustworthy — its null *exceeds* the signal); unbiased KID on the test split, mean ± sd over many 50-image subsets |
| `defect_feature_distance.py` | paired Inception feature distance between each annotated defect crop and the SAME box in the degraded image, against three controls (shuffled correspondence, background patches, cross-defect scale). The falsification test: if `paired` is not clearly below `shuffled`, the boxes carry no consistent information and annotation transfer fails. Reports the retention ratio `paired / cross` with image-level bootstrap CIs |
| `verify_fog_box_ssim.py` | in-box SSIM between each clean test image and its fogged counterpart, with background-matched control windows — quantifies how much fog destroys in-box structure (worst case 0.185 vs median 0.618, disclosed in the paper as a failure case) |

## Usage

```bash
pip install torch torchvision scipy numpy pillow      # plus opencv for the SSIM tool

python measure_fid_kid.py                 # --device cpu to force CPU
python defect_feature_distance.py
python verify_fog_box_ssim.py
```

Each script prints a full table and writes a machine-readable JSON to `runs/`.

## Note on the generation pipeline

The scripts that *generate* the degraded families (HazeGen fog synthesis and
the physical operators) are not part of this repository during peer review.
The degradation parameters themselves are published in
[`configs/degradation_params.json`](../configs/degradation_params.json), and
the fidelity of the resulting benchmark can be verified end-to-end with the
tools here once the datasets are available.
