#!/usr/bin/env python3
"""Diagnose MR signal weakness at the meta-model level."""
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extreme_price_movements.training import compute_meta_target

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ARTIFACT_ID = "20260212_170000"
ART_DIR = os.path.join(DATA_ROOT, "artifacts", ARTIFACT_ID)

# Load datasets from labels/ parquets
datasets = {}
labels_dir = os.path.join(ART_DIR, "labels")
if os.path.isdir(labels_dir):
    for f in os.listdir(labels_dir):
        if f.startswith("train_") and f.endswith(".parquet"):
            key = f.replace(".parquet", "")
            datasets[key] = pd.read_parquet(os.path.join(labels_dir, f))
    print(f"Loaded {len(datasets)} training datasets from labels/")
else:
    print("No labels/ dir found")
    sys.exit(1)

def load_oof(side, kind, H):
    pq = os.path.join(ART_DIR, "oof", f"oof_{side}_{kind}_H{H}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)["oof_prob"].values
    return None

# === SECTION 1: Return distributions ===
print("\n" + "="*70)
print("1. RETURN DISTRIBUTIONS")
print("="*70)
for side in ["long", "short"]:
    for kind in ["mr", "tf"]:
        for H in [2, 4, 8]:
            key = f"train_{side}_{kind}_{H}"
            if key not in datasets: continue
            r = datasets[key]["__y_ret__"].values.astype(float)
            b = datasets[key]["__y_bin__"].values.astype(float)
            print(f"  {side}_{kind} H={H}: n={len(r)} mean={np.mean(r):.6f} std={np.std(r):.6f} "
                  f"skew={skew(r):.3f} pos={np.mean(b>=0.5):.3f}")

# === SECTION 2: Cross-horizon correlations ===
print("\n" + "="*70)
print("2. CROSS-HORIZON RETURN CORRELATIONS")
print("="*70)
for side in ["long", "short"]:
    for kind in ["mr", "tf"]:
        rets = {}
        for H in [2, 4, 8]:
            key = f"train_{side}_{kind}_{H}"
            if key in datasets:
                rets[H] = datasets[key]["__y_ret__"].values.astype(float)
        if len(rets) < 2: continue
        ml = min(len(v) for v in rets.values())
        print(f"  {side}_{kind}:")
        for h1 in [2, 4]:
            for h2 in [4, 8]:
                if h2 <= h1: continue
                rho, _ = spearmanr(rets[h1][:ml], rets[h2][:ml])
                print(f"    H{h1}↔H{h2}: rho={rho:.4f}")

# === SECTION 3: Meta target quality ===
print("\n" + "="*70)
print("3. META TARGET — HORIZON CONTRIBUTION")
print("="*70)
for side in ["long", "short"]:
    for kind in ["mr", "tf"]:
        rets = {}
        for H in [2, 4, 8]:
            key = f"train_{side}_{kind}_{H}"
            if key in datasets:
                rets[H] = datasets[key]["__y_ret__"].values.astype(float)
        if not all(h in rets for h in [2,4,8]): continue
        ml = min(len(v) for v in rets.values())
        r2, r4, r8 = rets[2][:ml], rets[4][:ml], rets[8][:ml]
        mt = compute_meta_target(r2, r4, r8)
        lr2 = np.log1p(np.clip(r2, -0.999, None))
        lr4 = np.log1p(np.clip(r4, -0.999, None))
        lr8 = np.log1p(np.clip(r8, -0.999, None))
        c2 = float(np.corrcoef(lr2, mt)[0,1]) if np.std(lr2)>1e-9 else 0
        c4 = float(np.corrcoef(lr4, mt)[0,1]) if np.std(lr4)>1e-9 else 0
        c8 = float(np.corrcoef(lr8, mt)[0,1]) if np.std(lr8)>1e-9 else 0
        h2h8, _ = spearmanr(r2, r8)
        print(f"  {side}_{kind}: target_std={np.std(mt):.6f} "
              f"corr_H2={c2:.3f} corr_H4={c4:.3f} corr_H8={c8:.3f} H2↔H8={h2h8:.4f}")
        if h2h8 < 0:
            print(f"    ⚠️  H2↔H8 NEGATIVE — H8 dilutes signal!")

# === SECTION 4: OOF dispersion ===
print("\n" + "="*70)
print("4. OOF PREDICTION ↔ RETURN CORRELATION")
print("="*70)
for side in ["long", "short"]:
    for kind in ["mr", "tf"]:
        for H in [2, 4, 8]:
            oof = load_oof(side, kind, H)
            if oof is None: continue
            key = f"train_{side}_{kind}_{H}"
            if key not in datasets: continue
            n = min(len(oof), len(datasets[key]))
            yr = datasets[key]["__y_ret__"].values[:n].astype(float)
            o = oof[:n]
            if np.std(o) < 1e-9: continue
            rc = float(np.corrcoef(o, yr)[0,1])
            rs, _ = spearmanr(o, yr)
            print(f"  {side}_{kind} H={H}: oof_std={np.std(o):.4f} "
                  f"oof↔ret Pearson={rc:.4f} Spearman={rs:.4f}")

# === SECTION 5: Alternative MR target ===
print("\n" + "="*70)
print("5. ALTERNATIVE META TARGETS FOR MR")
print("="*70)
for side in ["long", "short"]:
    kind = "mr"
    rets = {}
    for H in [2, 4, 8]:
        key = f"train_{side}_{kind}_{H}"
        if key in datasets:
            rets[H] = datasets[key]["__y_ret__"].values.astype(float)
    if not all(h in rets for h in [2,4,8]): continue
    ml = min(len(v) for v in rets.values())
    r2, r4, r8 = rets[2][:ml], rets[4][:ml], rets[8][:ml]
    lr2 = np.log1p(np.clip(r2, -0.999, None)).astype(np.float32)
    lr4 = np.log1p(np.clip(r4, -0.999, None)).astype(np.float32)
    lr8 = np.log1p(np.clip(r8, -0.999, None)).astype(np.float32)
    
    targets = {
        "current(0.40/0.35/0.25)": compute_meta_target(r2, r4, r8),
        "mr_heavy(0.60/0.30/0.10)": 0.60*lr2 + 0.30*lr4 + 0.10*lr8,
        "h2_only": lr2,
        "h2h4_only(0.60/0.40)": 0.60*lr2 + 0.40*lr4,
    }
    
    oof = load_oof(side, kind, 2)  # Use H=2 OOF as proxy
    if oof is None: continue
    n = min(len(oof), ml)
    o = oof[:n]
    k10 = max(1, int(0.10 * n))
    top = np.argsort(o)[-k10:]
    bot = np.argsort(o)[:k10]
    
    print(f"\n  {side}_{kind} (using H=2 OOF, n={n}):")
    for name, t in targets.items():
        t = t[:n]
        spread = float(np.mean(t[top]) - np.mean(t[bot]))
        rho, _ = spearmanr(o, t)
        print(f"    {name:30s}: spread={spread:.6f} IC={rho:.4f}")

print("\nDone.")
