# -*- coding: utf-8 -*-
"""
Evaluate the DANN / Advent checkpoints with the SAME protocol as every other
number in the paper, so the comparison against the arms is apples-to-apples.

WHY A DEDICATED LOADER
  train_da.py saves a bare `{"model": state_dict}` because it builds the network
  itself (a DetectionModel constructed from the yolo11n yaml with nc=7). Ultralytics'
  YOLO(...) loader expects a full checkpoint dict, so the model has to be rebuilt
  the same way and the weights loaded into it. Rebuilding from the SAME yaml with
  the SAME nc reproduces the architecture exactly; anything else would silently
  drop or mis-shape tensors.

WHAT IT REPORTS
  1. A 5x5 table: each DA checkpoint evaluated on clean / fog / motion / low /
     composite TEST splits, AP@50 and AP@50:95, via the Ultralytics validator
     (pycocotools-backed) -- identical to the Ultralytics validator used everywhere else.
  2. The delta against the clean-only YOLO 11 baseline.
  3. Optionally a proxy A-distance and an RBF MMD between the clean and degraded
     feature distributions, as an additional domain-shift measurement. Both are
     computed on the TRAIN split with a held-out classifier, so the number is a
     domain-shift measurement, not a re-use of the training pairs.

Usage
  python eval_da.py --ckpt runs/dann_seed0/weights/best.pt --tag dann
  python eval_da.py --ckpt runs/dann_seed0/weights/best.pt --tag dann --a-distance
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import sys as _sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from _paths import ROOT  # repository root; override with EWTVD_ROOT
WS = ROOT

WS = ROOT
EXT = WS / "datasets" / "Extreme-WTVD"
CLEAN = WS / "datasets" / "WTVD"
RUNS_ARMS = WS / "runs"  # directory holding the clean-trained YOLO 11 runs
OUT = _HERE.parent / "detection" / "da_baselines" / "runs"
WEIGHTS = _HERE.parent / "detection" / "baselines" / "weights" / "yolo11n.pt"
NAMES = ['burning', 'crack', 'deformity', 'dirt', 'oil', 'peeling', 'rusty']
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

EVAL_SETS = {
    "clean": CLEAN,
    "fog": EXT / "fog",
    "motion": EXT / "motion",
    "low": EXT / "low",
    "composite": EXT / "composite",
}

LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def build_eval_yaml(root, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    y = out_dir / f"{root.name}.yaml"
    y.write_text(
        f"path: {str(root).replace(chr(92), '/')}\n"
        f"train: test/images\nval: test/images\ntest: test/images\n\n"
        "nc: 7\n"
        "names: ['burning', 'crack', 'deformity', 'dirt', 'oil', "
        "'peeling', 'rusty']\n", encoding="utf-8")
    return y


def load_da_model(ckpt_path, device):
    """Rebuild the nc=7 yolo11n DetectionModel and load the DA state dict.

    The architecture is reconstructed from the pretrained yolo11n checkpoint's
    own yaml (so neck/head wiring is whatever the framework ships) with nc set to
    the 7 WTVD classes -- exactly what train_da.py did.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    ref = YOLO(str(WEIGHTS))                 # gives us args, stride, names, device
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model" in ckpt and not isinstance(ckpt["model"], dict):
        # a full Ultralytics checkpoint (e.g. an arm) -- use it directly, and
        # prefer the EMA copy exactly as the framework's own evaluation does
        log("  note: full Ultralytics checkpoint; using the EMA weights")
        return (ckpt.get("ema") or ckpt["model"]).to(device).eval()

    sd_da = ckpt["model"]
    cfg = dict(ref.model.yaml)
    cfg["nc"] = len(NAMES)
    net = DetectionModel(cfg, nc=len(NAMES), verbose=False)

    sd = net.state_dict()
    matched = miss = 0
    for k, v in sd.items():
        if k in sd_da and sd_da[k].shape == v.shape:
            sd[k] = sd_da[k]
            matched += 1
        else:
            miss += 1
    net.load_state_dict(sd)
    log(f"  weights loaded: {matched}/{len(sd)} tensors matched, "
        f"{miss} left at init")

    net.names = {i: n for i, n in enumerate(NAMES)}
    net.nc = len(NAMES)
    net.args = ref.model.args
    if hasattr(ref.model, "stride"):
        net.stride = ref.model.stride
    net.to(device).eval()
    return net, ref


def evaluate(ckpt_path, tag):
    """Score one checkpoint on all five test sets."""
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 100)
    log(f"DA CHECKPOINT EVALUATION -- {tag}")
    log(f"  ckpt: {ckpt_path}")
    log("=" * 100)

    net, ref = load_da_model(ckpt_path, device)
    ref.model = net
    ref.overrides["task"] = "detect"

    cfg_dir = OUT / f"_evalcfgs_{tag}"
    result = {}
    for sname, root in EVAL_SETS.items():
        y = build_eval_yaml(root, cfg_dir)
        try:
            m = ref.val(data=str(y), split="test", imgsz=640, verbose=False,
                        plots=False, save_json=False)
            d = m.results_dict
            result[sname] = {
                "ap50": float(d.get("metrics/mAP50(B)", 0)) * 100,
                "ap5095": float(d.get("metrics/mAP50-95(B)", 0)) * 100,
                "precision": float(d.get("metrics/precision(B)", 0)) * 100,
                "recall": float(d.get("metrics/recall(B)", 0)) * 100,
            }
            log(f"  {sname:10s} P={result[sname]['precision']:5.2f} "
                f"R={result[sname]['recall']:5.2f} "
                f"AP50={result[sname]['ap50']:5.2f} "
                f"AP50:95={result[sname]['ap5095']:5.2f}")
        except Exception as e:
            result[sname] = None
            log(f"  {sname:10s} FAILED {str(e)[:150]}")

    return result


def compare_to_clean_arm(result):
    """Delta vs the clean-only arm, the baseline the letter quotes."""
    mj = RUNS_ARMS / "yolo11n_clean_seed0" / "results.json"
    if not mj.is_file():
        log("\n  (clean-trained YOLO 11 run not found -- skipping the delta table)")
        return
    arm = json.loads(mj.read_text(encoding="utf-8")).get("1_clean", {})
    if not arm:
        log("\n  (1_clean row absent -- skipping the delta table)")
        return
    log("\n" + "=" * 100)
    log("DELTA vs the clean-only arm (1_clean), AP@50:95 points")
    log("=" * 100)
    hdr = f"{'test set':<12s}{'1_clean':>12s}{'this ckpt':>12s}{'delta':>12s}"
    log(hdr)
    log("-" * len(hdr))
    for s in EVAL_SETS:
        b = arm.get(s)
        a = result.get(s)
        if not b or not a:
            log(f"{s:<12s}{'--':>12s}{'--':>12s}{'--':>12s}")
            continue
        log(f"{s:<12s}{b['ap5095']:>12.2f}{a['ap5095']:>12.2f}"
            f"{a['ap5095'] - b['ap5095']:>+12.2f}")


# ---------------------------------------------------------------------------
# proxy A-distance and MMD
# ---------------------------------------------------------------------------
def extract_features(net, img_dir, n_max, imgsz=640, device="cuda"):
    """Pooled multi-scale features for up to n_max images."""
    import cv2

    tap_idx = list(getattr(net.model[-1], "f", [16, 19, 22]))
    feats_out = []

    def make_hook(store):
        def hook(_m, _i, out):
            store.append(out)
        return hook

    store = []
    hs = [net.model[i].register_forward_hook(make_hook(store)) for i in tap_idx]
    files = sorted(p for p in Path(img_dir).glob("*")
                   if p.suffix.lower() in IMG_EXTS)[:n_max]
    with torch.no_grad():
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            im = cv2.resize(im, (imgsz, imgsz))
            x = torch.from_numpy(im[:, :, ::-1].copy()).permute(
                2, 0, 1).float().div(255).unsqueeze(0).to(device)
            store.clear()
            net(x)
            pooled = [torch.nn.functional.adaptive_avg_pool2d(
                s, 1).flatten(1)[0].float().cpu() for s in store]
            feats_out.append(torch.cat(pooled).numpy())
    for h in hs:
        h.remove()
    return np.stack(feats_out) if feats_out else np.zeros((0, 1))


def rbf_mmd(x, y, gammas=(0.25, 1.0, 4.0)):
    """Unbiased MMD^2 with a mixture of RBF bandwidths (multi-kernel MMD)."""
    def k(a, b, g):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-g * d)
    m = min(len(x), len(y))
    x, y = x[:m], y[:m]
    val = 0.0
    for g in gammas:
        kxx, kyy, kxy = k(x, x, g), k(y, y, g), k(x, y, g)
        n = m
        a = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        b = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        val += a + b - 2 * kxy.mean()
    return float(val / len(gammas))


def proxy_a_distance(x, y, seed=0):
    """Proxy A-distance = 2(1 - 2*eps) per Ben-David et al.

    eps is the held-out error of a domain classifier trained to separate the two
    feature distributions, so A-distance grows with how separable they are.
    Implemented as an L2-regularised logistic regression in torch: the tia
    environment has no scikit-learn, and a dependency-free implementation keeps
    the number reproducible.
    """
    torch.manual_seed(seed)
    X = np.vstack([x, y]).astype(np.float64)
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))].astype(np.float64)

    # stratified 60/40 split
    rng = np.random.RandomState(seed)
    idx0 = rng.permutation(np.flatnonzero(lab == 0))
    idx1 = rng.permutation(np.flatnonzero(lab == 1))
    cut0, cut1 = int(0.6 * len(idx0)), int(0.6 * len(idx1))
    tr = np.r_[idx0[:cut0], idx1[:cut1]]
    te = np.r_[idx0[cut0:], idx1[cut1:]]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(lab[tr], dtype=torch.float32)
    yte = torch.tensor(lab[te], dtype=torch.float32)

    lin = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(lin(Xtr).squeeze(1), ytr) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = (lin(Xte).squeeze(1) > 0).float()
        err = float((pred != yte).float().mean())
    return 2.0 * (1.0 - 2.0 * err), err


def a_distance_block(net, n=600, device="cuda"):
    log("\n" + "=" * 100)
    log("DOMAIN-SHIFT QUANTIFICATION (proxy A-distance and multi-kernel MMD)")
    log("=" * 100)
    log("Features: pooled neck activations. Source = clean TRAIN images,")
    log("target = each degraded TRAIN family. A classifier is trained on a 60%")
    log("split and scored on the held-out 40%, so eps is a genuine generalisation")
    log("error and A-distance = 2(1 - 2*eps) is not fitted to itself.")
    log("")
    src = extract_features(net, CLEAN / "train" / "images", n, device=device)
    log(f"  source features: {src.shape}")
    hdr = f"{'target':<12s}{'eps':>10s}{'A-dist':>10s}{'MMD^2':>12s}"
    log("")
    log(hdr)
    log("-" * len(hdr))
    out = {}
    for fam in ("fog", "motion", "low", "composite"):
        tgt = extract_features(net, EXT / fam / "train" / "images", n,
                               device=device)
        if len(tgt) == 0:
            log(f"{fam:<12s}{'--':>10s}{'--':>10s}{'--':>12s}")
            continue
        ad, eps = proxy_a_distance(src, tgt)
        mmd = rbf_mmd(src, tgt)
        out[fam] = {"a_distance": ad, "clf_error": eps, "mmd": mmd}
        log(f"{fam:<12s}{eps:>10.3f}{ad:>10.3f}{mmd:>12.5f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--a-distance", action="store_true")
    ap.add_argument("--n-feat", type=int, default=600)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = _HERE.parent / "detection" / "da_baselines" / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {"tag": args.tag, "ckpt": str(ckpt)}

    if not args.skip_eval:
        res = evaluate(ckpt, args.tag)
        payload["eval"] = res
        compare_to_clean_arm(res)

    if args.a_distance:
        # load_da_model returns a bare model for a full Ultralytics checkpoint and
        # a (net, reference) tuple for a train_da.py checkpoint; accept both.
        loaded = load_da_model(ckpt, device)
        net = loaded[0] if isinstance(loaded, tuple) else loaded
        # full Ultralytics checkpoints carry FP16 weights (the framework stores the
        # EMA in half precision); the feature extractor feeds float32, so normalise
        # the dtype here rather than at every activation tap.
        net = net.float()
        payload["domain_shift"] = a_distance_block(net, args.n_feat, device)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval_{args.tag}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / f"eval_{args.tag}.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nwritten: da_baselines/runs/eval_{args.tag}.json , .txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
