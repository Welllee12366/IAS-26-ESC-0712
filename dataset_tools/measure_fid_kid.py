# -*- coding: utf-8 -*-
"""
Measure image-fidelity of the synthesized Extreme-WTVD benchmark.

Reports FID and KID between the clean WTVD images and each degraded family,
using an ImageNet-pretrained Inception-v3 (2048-d pool3) -- a backbone that
was NEVER trained on WTVD, so it cannot have been tuned to like the synthetic
data.

WHY THIS SCRIPT LOOKS THE WAY IT DOES
-------------------------------------
A naive "FID per condition on the 177-image test split" is not trustworthy, and
finding that out is half the point:

  * Calibrating FID against its own null gave
        FID(half of clean test, other half of clean test) = 136.2
    which EXCEEDS the value for motion blur (60.4), low-light (64.5) and
    composite (92.3). A statistic whose null exceeds the signal supports no
    conclusion. Cause: finite-sample bias -- at n = 88 in 2048 dimensions the
    covariance estimate is badly rank-deficient and the sqrtm term is noise.

  * KID's estimator is unbiased, so its null is ~0.4e-3 against a signal of
    3-26e-3 and it stays valid at small n, provided many small subsets are used.

The script therefore reports the metric in three ways, all of which are needed
to make an honest claim:

  PART A  FID on the TRAINING split (3577 images per condition), where a 2048-d
          covariance is estimable. A disjoint half-split null calibrates it.
  PART B  FID on the TEST split (n=177) after PCA-reducing the pooled features,
          at several retained dimensions, each row shown WITH its own null so
          the reader can see which row is trustworthy.
  PART C  KID on the TEST split, mean +/- sd over many 50-image subsets (the
          standard KID convention), plus its null.

Output: runs/fid_kid_final.json  (machine-readable, all three parts).

Usage
  python measure_fid_kid.py
  python measure_fid_kid.py --device cpu --out my_fid.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg

WS = Path(__file__).resolve().parents[1]   # repository root
CLEAN = WS / "datasets" / "WTVD"
EXT = WS / "datasets" / "Extreme-WTVD"
CONDS = ["fog", "motion", "low", "composite"]
SEED = 0
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def uniq_images(d):
    """Case-insensitive de-duplication of a folder listing.

    On Windows `glob('*.jpg')` and `glob('*.JPG')` return the same files; a
    naive enumeration would therefore count every image twice and bias the
    covariance. Keeping one entry per lower-cased name avoids that bug.
    """
    seen = {}
    for p in sorted(Path(d).iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            seen.setdefault(p.name.lower(), p)
    return [seen[k] for k in sorted(seen)]


@torch.no_grad()
def feats(model, files, dev, batch=32):
    """Inception-v3 pool3 features with the standard FID preprocessing."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(files), batch):
        ims = []
        for p in files[i:i + batch]:
            im = Image.open(p).convert("RGB").resize((299, 299), Image.BICUBIC)
            ims.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
        x = torch.stack(ims).float().to(dev) / 255.0
        out.append(model((x - mean) / std).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fid(a, b):
    """Frechet Inception Distance (Fréchet distance between two Gaussians)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    d = mu1 - mu2
    s, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if np.iscomplexobj(s):
        s = s.real
    return float(d.dot(d) + np.trace(c1 + c2 - 2 * s))


def kid(a, b, degree=3, coef=1.0):
    """Unbiased polynomial-kernel MMD^2 (the KID estimator)."""
    gamma = 1.0 / a.shape[1]
    K = lambda X, Y: (gamma * X.dot(Y.T) + coef) ** degree
    m = a.shape[0]
    kxx, kyy, kxy = K(a, a), K(b, b), K(a, b)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1))
                 - 2.0 * kxy.mean())


def load_backbone(dev):
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(WS / "runs" / "fid_kid_final.json"))
    ap.add_argument("--n-subset", type=int, default=50,
                    help="KID subset size (standard default 50)")
    ap.add_argument("--n-draws", type=int, default=100,
                    help="number of KID subset draws (standard default 100)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    m = load_backbone(dev)
    rng = np.random.default_rng(SEED)
    report = {}

    # ================= PART A: training split, full sample size =============
    print("=" * 96)
    print("PART A -- FID on the TRAINING split (large n, 2048-d covariance estimable)")
    print("=" * 96)
    cf = uniq_images(CLEAN / "train" / "images")
    fc = feats(m, cf, dev)
    print(f"clean train: {len(cf)} images -> features {fc.shape}")

    idx = rng.permutation(len(fc))
    nullA = fid(fc[idx[:1000]], fc[idx[1000:2000]])
    print(f"\nNULL: FID(clean_a, clean_b) at n=1000 each = {nullA:.3f}")
    print("      (must be far below the condition values for them to be meaningful)")

    print(f"\n{'condition':<12}{'n':>6}{'FID':>10}{'KID x1e3':>11}")
    print("-" * 39)
    for cond in CONDS:
        d = EXT / cond / "train" / "images"
        if not d.is_dir():
            continue
        files = uniq_images(d)
        fg = feats(m, files, dev)
        n = min(len(fc), len(fg), 2000)
        ia = rng.choice(len(fc), n, replace=False)
        ib = rng.choice(len(fg), n, replace=False)
        f = fid(fc[ia], fg[ib])
        k = float(np.mean([
            kid(fc[rng.choice(len(fc), args.n_subset, replace=False)],
                fg[rng.choice(len(fg), args.n_subset, replace=False)])
            for _ in range(50)]))
        print(f"{cond:<12}{len(files):>6}{f:>10.2f}{k*1e3:>11.4f}")
        report.setdefault(cond, {})["train"] = dict(
            n=len(files), fid=float(f), kid=float(k), fid_null_1000=float(nullA),
            subset=n)
    report["_null_train"] = float(nullA)

    # ================= PART B: test split, PCA-reduced ======================
    print("\n" + "=" * 96)
    print("PART B -- FID on the TEST split (n=177) with PCA-reduced features")
    print("=" * 96)
    cft = uniq_images(CLEAN / "test" / "images")
    fct = feats(m, cft, dev)

    all_f = [fct]
    cond_f = {}
    for cond in CONDS:
        d = EXT / cond / "test" / "images"
        files = uniq_images(d)
        fg = feats(m, files, dev)
        cond_f[cond] = fg
        all_f.append(fg)
    # PCA basis fitted on the pooled clean+generated test features
    pooled = np.vstack(all_f)
    mu = pooled.mean(0)
    _, _, Vt = np.linalg.svd(pooled - mu, full_matrices=False)

    for k in (32, 64, 128):
        P = Vt[:k].T
        ca = (fct - mu).dot(P)
        idx = rng.permutation(len(ca))
        nullk = fid(ca[idx[:80]], ca[idx[80:]])
        line = f"  k={k:3d} null={nullk:8.2f}  |"
        vals = {}
        for cond in CONDS:
            g = (cond_f[cond] - mu).dot(P)
            v = fid(ca, g)
            vals[cond] = v
            line += f" {cond}={v:7.2f}"
        print(line)
        report.setdefault("_pca_test", {})[str(k)] = dict(null=float(nullk), **vals)

    print("\n  The null must be well below the condition values. Increasing the")
    print("  retained dimension lowers the signal fidelity; the row where the null")
    print("  is smallest relative to the signal is the trustworthy one.")

    # ================= PART C: KID at test size (unbiased) ==================
    print("\n" + "=" * 96)
    print("PART C -- KID on the TEST split (unbiased, valid at n=177)")
    print("=" * 96)
    idx = rng.permutation(len(fct))
    ha, hb = fct[idx[:88]], fct[idx[88:]]
    nullk = float(np.mean([
        kid(ha[rng.choice(88, args.n_subset, replace=False)],
            hb[rng.choice(88, args.n_subset, replace=False)])
        for _ in range(args.n_draws)]))
    print(f"  NULL: KID(clean_a, clean_b) = {nullk*1e3:.4f} x1e-3")
    print(f"\n  {'condition':<12}{'KID x1e3':>11}{'sd':>9}{'95% CI':>20}")
    print("  " + "-" * 50)
    for cond in CONDS + ["combined"]:
        d = EXT / cond / "test" / "images"
        if not d.is_dir():
            continue
        files = uniq_images(d)
        fg = cond_f.get(cond) if cond in cond_f else feats(m, files, dev)
        ks = [kid(fct[rng.choice(len(fct), args.n_subset, replace=False)],
                  fg[rng.choice(len(fg), args.n_subset, replace=False)])
              for _ in range(args.n_draws)]
        mean, sd = float(np.mean(ks)), float(np.std(ks))
        lo, hi = np.percentile(ks, [2.5, 97.5])
        ci = "[{:.2f}, {:.2f}]".format(lo * 1e3, hi * 1e3)
        print(f"  {cond:<12}{mean*1e3:>11.4f}{sd*1e3:>9.4f}{ci:>20}")
        report.setdefault(cond, {})["test_kid"] = dict(
            mean=mean, sd=sd, ci95=[float(lo), float(hi)], null=nullk)
    report["_null_kid_test"] = nullk

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
