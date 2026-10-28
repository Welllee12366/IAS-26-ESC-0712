# -*- coding: utf-8 -*-
"""
Defect-region feature distance: does defect geometry survive synthesis?

WHY PAIRED DISTANCES AND NOT ANOTHER FID
  A useful check is a feature distance restricted to the annotated boxes.
  FID is the wrong instrument here for two reasons: the test split has 177
  images, so a 2048-d covariance is rank-deficient (the earlier null calibration
  showed FID is unusable at that size), and restricting to boxes leaves only a
  few hundred crops, which is smaller still. A PAIRED distance has no covariance
  to estimate -- each clean crop is compared with the crop of the SAME box in the
  degraded image -- so it is stable at this sample size.

WHAT IS MEASURED
  Encoder: ImageNet InceptionV3 (2048-d, final pooling). Frozen, and never
  trained on WTVD, so it cannot have been tuned to like the synthetic data.

  For each condition c in {fog, motion, low, composite}:

    paired      D( f(clean box i), f(degraded box i) )
                the same annotated defect, only the imaging condition changed.
                THIS is the quantity that should be small if synthesis
                preserves the defect.

    background  the same paired distance, but for size-matched patches placed
                where there is NO annotation (IoU 0 against every box in the
                image). This is the control that parallels the SSIM analysis
                already in the benchmark README, and it is what makes the
                defect number interpretable: if the paired distance is
                SMALLER for defect regions than for background, then the
                degradation is not selectively destroying the objects we want
                to detect.

    cross       D( f(clean box i), f(clean box j) ), i != j, area-matched.
                The natural scale: how far apart two UNRELATED defects sit.
                A paired distance near zero relative to this means defect
                identity dominates the degradation.

    shuffled    D( f(clean box i), f(degraded box j) ), i != j, area-matched.
                Same distribution as `paired` but with a deliberately broken
                correspondence. If paired is not clearly below shuffled then
                the boxes carry no consistent information and the whole
                transfer argument fails -- so this is the falsification test
                for the measurement itself.

  Reported as a ratio, retention = paired / cross. Sources are area-matched
  (log-area decile) in every comparison, because feature distance grows with
  crop content and scale mismatch would confound the result.

Outputs runs/defect_feature_distance.json and a printed table.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WS = Path(__file__).resolve().parents[1]   # repository root
CLEAN = WS / "datasets" / "WTVD"
EXT = WS / "datasets" / "Extreme-WTVD"
CONDS = ["fog", "motion", "low", "composite"]
SUFFIX = {"fog": "", "motion": "__motion0", "low": "__low0",
          "composite": "__composite0"}
EXTMAP = {"fog": ".png"}
SEED = 0
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def degraded_path(cond, stem):
    """Locate the degraded counterpart.

    Naming is NOT uniform: the physical families append a variant suffix
    (`{stem}__motion0.jpg`) while the generative fog family keeps the stem and
    changes the extension. A single `{stem}.jpg` lookup silently matched only
    fog, which is the bug that previously produced a zero-pair dataset.
    """
    d = EXT / cond / "test" / "images"
    suf = SUFFIX[cond]
    ext = EXTMAP.get(cond, ".jpg")
    p = d / (stem + suf + ext)
    if p.is_file():
        return p
    for q in d.glob(stem + "*"):
        return q
    return None


def load_labels(p):
    """YOLO labels -> list of (cls, cx, cy, w, h) normalised."""
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) >= 5:
            out.append(tuple(float(v) for v in t[:5]))
    return out


def crop(img, cx, cy, w, h, pad=0.20, out=299):
    """Crop a normalised box with padding, clamped to the image, squared off."""
    W, H = img.size
    bw, bh = w * W, h * H
    # pad, and then square off, so the encoder sees the same aspect ratio for
    # every crop and a thin box is not resampled to a sliver
    side = max(bw, bh) * (1.0 + 2 * pad)
    cx_px, cy_px = cx * W, cy * H
    x0 = int(round(cx_px - side / 2))
    y0 = int(round(cy_px - side / 2))
    side = int(round(side))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = min(W, x0 + side)
    y1 = min(H, y0 + side)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img.crop((x0, y0, x1, y1)).resize((out, out), Image.BICUBIC)


def random_bg_patch(img, boxes, w, h, rng, tries=60):
    """A size-matched patch with IoU 0 against every annotation."""
    W, H = img.size
    bw, bh = w * W, h * H
    for _ in range(tries):
        cx = rng.uniform(bw / 2 / W, 1 - bw / 2 / W) if bw < W else 0.5
        cy = rng.uniform(bh / 2 / H, 1 - bh / 2 / H) if bh < H else 0.5
        # reject anything overlapping an annotation
        if all(abs(cx - b[1]) >= (bw + b[3] * W) / 2 / W
               or abs(cy - b[2]) >= (bh + b[4] * H) / 2 / H
               for b in boxes):
            return cx, cy, w, h
    return None


def area_bucket(w, h, nb=10):
    a = max(w * h, 1e-9)
    return min(nb - 1, int((np.log10(a) + 5.0) / 5.0 * nb))


@torch.no_grad()
def encode(model, crops, dev, batch=32):
    if not crops:
        return np.zeros((0, 2048))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), batch):
        ims = np.stack([np.array(c.convert("RGB")) for c in crops[i:i + batch]])
        x = torch.from_numpy(ims).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - mean) / std).to(dev)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def cos_dist(a, b):
    """Per-row cosine distance between two feature matrices of equal length."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (a * b).sum(1)


def cluster_bootstrap(vals_a, ids_a, vals_b, ids_b, n_boot=2000, seed=0):
    """Bootstrap CIs that resample IMAGES, not crops.

    Defects inside one photograph share lighting, background and viewpoint, so
    crops are not independent and a crop-level bootstrap would understate the
    interval. Resampling whole images keeps the within-image correlation intact.
    `vals_b`/`ids_b` may be None when only one quantity is being summarised.
    """
    rng = np.random.default_rng(seed)
    img_ids = np.unique(ids_a)
    if len(img_ids) < 5:
        return None
    idx_by_img = {g: np.flatnonzero(ids_a == g) for g in img_ids}
    idx_by_img_b = ({g: np.flatnonzero(ids_b == g) for g in img_ids}
                    if vals_b is not None else None)
    ma, mb, md = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(img_ids, len(img_ids), replace=True)
        ia = np.concatenate([idx_by_img[g] for g in pick])
        ma.append(float(np.mean(vals_a[ia])))
        if vals_b is not None:
            ib = np.concatenate([idx_by_img_b[g] for g in pick])
            mb.append(float(np.mean(vals_b[ib])))
            # only meaningful when the two are paired row-for-row
            if len(ib) == len(ia):
                md.append(float(np.mean(vals_a[ia] - vals_b[ib])))
    out = {"mean": float(np.mean(vals_a)),
           "ci95": [float(np.percentile(ma, 2.5)),
                    float(np.percentile(ma, 97.5))]}
    if vals_b is not None and mb:
        out["b_mean"] = float(np.mean(vals_b))
        out["b_ci95"] = [float(np.percentile(mb, 2.5)),
                         float(np.percentile(mb, 97.5))]
        # paired difference, valid only when the two arrays are aligned
        if md:
            out["diff"] = float(np.mean(vals_a - vals_b))
            out["diff_ci95"] = [float(np.percentile(md, 2.5)),
                                float(np.percentile(md, 97.5))]
            out["diff_excludes_zero"] = bool(
                np.percentile(md, 2.5) > 0 or np.percentile(md, 97.5) < 0)
    out["n_images"] = int(len(img_ids))
    return out


def matched_partner(pool, i, rng):
    """An index j != i drawn from the caller's area+class matched pool."""
    cand = [j for j in pool if j != i]
    return rng.choice(cand) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so a running training job keeps the GPU")
    ap.add_argument("--n-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    dev = torch.device(args.device)
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval().to(dev)

    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    img_dir = CLEAN / "test" / "images"
    lab_dir = CLEAN / "test" / "labels"
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if args.n_images:
        files = files[:args.n_images]

    log("=" * 100)
    log("DEFECT-REGION FEATURE DISTANCE (frozen ImageNet InceptionV3)")
    log("=" * 100)
    log(f"  clean test images: {len(files)}")

    # ---- pass 1: collect every clean crop, each with its provenance ---------
    clean_crops, meta, bg_crops = [], [], []
    img_of_defect, img_of_bg = [], []
    per_cond_crops = {c: [] for c in CONDS}
    per_cond_bg = {c: [] for c in CONDS}
    nobj = 0
    for fid, f in enumerate(files):
        stem = f.stem
        boxes = load_labels(lab_dir / (stem + ".txt"))
        if not boxes:
            continue
        im_clean = Image.open(f).convert("RGB")
        ims_deg = {}
        for c in CONDS:
            p = degraded_path(c, stem)
            if p is not None:
                try:
                    ims_deg[c] = Image.open(p).convert("RGB")
                except Exception:
                    pass
        for (cls, cx, cy, w, h) in boxes:
            cc = crop(im_clean, cx, cy, w, h)
            if cc is None:
                continue
            nobj += 1
            meta.append(dict(stem=stem, cls=int(cls), w=w, h=h))
            clean_crops.append(cc)
            img_of_defect.append(fid)
            for c in CONDS:
                if c in ims_deg:
                    dc = crop(ims_deg[c], cx, cy, w, h)
                    if dc is not None:
                        per_cond_crops[c].append(dc)
                    else:
                        per_cond_crops[c].append(None)
                else:
                    per_cond_crops[c].append(None)
            # size-matched background patch, same coordinates in every image
            #
            # bg_crops and per_cond_bg are appended together, always, so row i
            # of one corresponds to row i of the other; a skipped patch is
            # recorded as None rather than omitted, which is what keeps them in
            # step.
            spot = random_bg_patch(im_clean, boxes, w, h, rng)
            bc = crop(im_clean, *spot) if spot is not None else None
            bg_crops.append(bc)
            img_of_bg.append(fid)
            for c in CONDS:
                if bc is not None and c in ims_deg:
                    per_cond_bg[c].append(crop(ims_deg[c], *spot))
                else:
                    per_cond_bg[c].append(None)

    log(f"  annotated defects available : {nobj}")
    if nobj < 30:
        log("  too few defects to measure -- aborting")
        return 1

    fc = encode(m, clean_crops, dev)
    log(f"  clean defect features       : {fc.shape}")
    # bg_crops has None for patches that could not be placed, so remember which
    # rows were actually encoded -- fb's row r corresponds to bg_rows[r].
    bg_rows = [i for i, c in enumerate(bg_crops) if c is not None]
    fb = encode(m, [bg_crops[i] for i in bg_rows], dev)
    fb_pos = {i: r for r, i in enumerate(bg_rows)}
    log(f"  background control patches  : {fb.shape}")

    # area buckets over the DEFECT boxes, for scale matching
    keys = []
    for md in meta:
        keys.append(f"{area_bucket(md['w'], md['h'])}-{md['cls']}")
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)

    report = {}

    def partner(i):
        return matched_partner(buckets[keys[i]], i, rng)

    # ---- cross / shuffled references, computed once, shared by all conds ----
    ci, cj = [], []
    for i in range(nobj):
        j = partner(i)
        if j is not None:
            ci.append(i)
            cj.append(j)
    ci, cj = np.array(ci), np.array(cj)
    cross_d = float(cos_dist(fc[ci], fc[cj]).mean()) if len(ci) else float("nan")
    log("")
    log(f"  cross-defect reference (clean vs a DIFFERENT clean defect, "
        f"area+class matched, n={len(ci)}): {cross_d:.4f}")

    hdr = (f"  {'condition':<12}{'paired':>10}{'shuffled':>10}{'background':>12}"
           f"{'cross':>9}{'paired/cross':>14}{'paired<shuf?':>14}")
    log("")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    for c in CONDS:
        deg = per_cond_crops[c]
        keep = [i for i in range(nobj) if deg[i] is not None]
        if len(keep) < 30:
            log(f"  {c:<12}{'-- insufficient pairs':>20}")
            continue
        fd = encode(m, [deg[i] for i in keep], dev)
        pd_row = cos_dist(fc[keep], fd)
        paired = float(pd_row.mean())
        ids = np.array([img_of_defect[i] for i in keep])

        # shuffled: clean i vs degraded j, same area+class, i != j
        idx_of = {g: p for p, g in enumerate(keep)}
        si, sj = [], []
        for i in keep:
            j = partner(i)
            if j is not None and j in idx_of:
                si.append(i)
                sj.append(idx_of[j])
        shuf_row = cos_dist(fc[np.array(si)], fd[np.array(sj)]) \
            if len(si) else np.zeros(0)
        shuf = float(shuf_row.mean()) if len(si) else float("nan")

        # background: same paired construction, non-annotated patches
        dbg = per_cond_bg[c]
        bk = [i for i in bg_rows if i < len(dbg) and dbg[i] is not None]
        background = float("nan")
        bg_row = np.zeros(0)
        bg_ids = np.zeros(0)
        if len(bk) >= 30:
            fdb = encode(m, [dbg[i] for i in bk], dev)
            fbc = fb[[fb_pos[i] for i in bk]]
            bg_row = cos_dist(fbc, fdb)
            bg_ids = np.array([img_of_bg[i] for i in bk])
            background = float(bg_row.mean())

        ratio = paired / cross_d
        ok = paired < shuf
        # Shuffle CI: the shuffled rows are drawn from `keep`, so their image
        # ids line up with `ids` only positionally; the CI for the shuffled
        # mean is therefore summarised on its own.
        ci_p = cluster_bootstrap(pd_row, ids, None, None)
        ci_s = (cluster_bootstrap(shuf_row,
                                  np.array([img_of_defect[i] for i in si]),
                                  None, None) if len(si) else None)
        ci_b = (cluster_bootstrap(bg_row, bg_ids, None, None)
                if len(bk) >= 30 else None)
        log(f"  {c:<12}{paired:>10.4f}{shuf:>10.4f}{background:>12.4f}"
            f"{cross_d:>9.4f}{ratio:>14.3f}{str(ok):>14}")
        if ci_p:
            log(f"    paired     95% CI [{ci_p['ci95'][0]:.4f}, "
                f"{ci_p['ci95'][1]:.4f}]  over {ci_p['n_images']} images")
        if ci_s:
            log(f"    shuffled   95% CI [{ci_s['ci95'][0]:.4f}, "
                f"{ci_s['ci95'][1]:.4f}]")
        if ci_b:
            log(f"    background 95% CI [{ci_b['ci95'][0]:.4f}, "
                f"{ci_b['ci95'][1]:.4f}]")
        report[c] = dict(paired=paired, shuffled=shuf, background=background,
                         cross=cross_d, ratio=ratio,
                         paired_below_shuffled=bool(ok), n_pairs=len(keep),
                         paired_ci95=ci_p["ci95"] if ci_p else None,
                         shuffled_ci95=ci_s["ci95"] if ci_s else None,
                         background_ci95=ci_b["ci95"] if ci_b else None,
                         paired_minus_shuffled=(
                             float(paired - shuf) if len(si) else None),
                         paired_minus_background=(
                             float(paired - background)
                             if len(bk) >= 30 else None))

    # ---- does paired sit BELOW shuffled, with the interval)?
    log("")
    log("  paired vs shuffled separation, using the two bootstrap intervals:")
    for c in CONDS:
        r = report.get(c)
        if not r or not r.get("shuffled_ci95") or not r.get("paired_ci95"):
            continue
        sep = r["paired_ci95"][1] < r["shuffled_ci95"][0]
        r["paired_shuffled_intervals_disjoint"] = bool(sep)
        log(f"    {c:<12}{'yes, disjoint' if sep else 'intervals overlap'}"
            f"   (paired max {r['paired_ci95'][1]:.4f} vs shuffled min "
            f"{r['shuffled_ci95'][0]:.4f})")


    report["_meta"] = dict(encoder="inception_v3_imagenet_2048",
                           clean_test_images=len(files), defect_crops=nobj,
                           seed=SEED)

    log("")
    log("=" * 100)
    log("READING THE TABLE")
    log("=" * 100)
    log("  paired < shuffled is the falsification test: if breaking the box")
    log("  correspondence does not increase the distance, the boxes carry no")
    log("  consistent information and the transfer claim fails.")
    log("  paired < background means the annotated defect regions are preserved")
    log("  at least as well as unannotated image content.")
    log("  paired/cross is the retention ratio: 0 would mean the degradation is")
    log("  completely defect-invariant, 1 means degraded defects are as far from")
    log("  their own clean version as from an unrelated defect.")

    (WS / "runs" / "defect_feature_distance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (WS / "runs" / "defect_feature_distance.txt").write_text(
        "\n".join(LOG), encoding="utf-8")
    log("\nwritten: runs/defect_feature_distance.json , .txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
