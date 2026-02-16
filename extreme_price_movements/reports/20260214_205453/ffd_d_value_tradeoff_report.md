# FFD d-value Trade-off Report (Cross-Asset)

## Scope

This report compares fixed-width FFD `d` values after implementing the remediation roadmap:
- log-space geometry features
- fixed close-only FFD (no adaptive ADF leakage)
- correctly named log returns (`lr_*`) with one-release compatibility aliases
- multi-d feature block for production and diagnostics

Reference artifacts:
- `extreme_price_movements/reports/20260214_205351/ffd_d_value_comparison.csv`
- `extreme_price_movements/reports/20260214_205453/ffd_d_value_comparison.csv`
- `extreme_price_movements/reports/20260214_205351/ffd_weight_window_sizes.csv`

---

## Experiment Matrix

### Run A — baseline diagnostic
- Label horizon: `H=24`
- Ridge alpha: `1.0`
- Purged CV folds: `5`
- Assets evaluated: `18`

### Run B — short-horizon sensitivity check
- Label horizon: `H=8`
- Ridge alpha: `0.5`
- Purged CV folds: `5`
- Assets evaluated: `18`

---

## Effective Memory / Compute by d

| d | K(d) | Warmup bars | Relative compute |
|---|---:|---:|---|
| 0.2 | 3382 | 3381 | highest |
| 0.3 | 2275 | 2274 | high |
| 0.4 | 1458 | 1457 | medium |
| 0.5 | 927 | 926 | low |
| 0.6 | 590 | 589 | lowest |

Implication: lower `d` carries much higher warmup and convolution cost.

---

## Aggregate Results (Event-Regime Focus)

### Run A (H=24, alpha=1.0)
Ranking by mean event IC-IR:
1. `d=0.4` → `ic_ir_event_mean=1.2558`
2. `d=0.3` → `1.2368`
3. `d=0.2` → `1.1844`
4. `d=0.5` → `1.1417`
5. `d=0.6` → `0.9776`

Additional context:
- Highest raw `ic_event_mean`: `d=0.2` (`0.0517`)
- Best risk-adjusted stability (IC-IR): `d=0.4`

### Run B (H=8, alpha=0.5)
Ranking by mean event IC-IR:
1. `d=0.6` → `ic_ir_event_mean=0.3752`
2. `d=0.5` → `0.3057`
3. `d=0.4` → `0.2272`
4. `d=0.3` → `0.0666`
5. `d=0.2` → `-0.1193`

Interpretation:
- For shorter horizons, higher `d` (faster/shorter memory) is favored.
- For longer horizons, mid/low `d` (`0.3-0.4`) is more robust on event IC-IR.

---

## Cross-Run Stability

Best-d stability across the two runs:
- Same best `d` for only `4/18` assets (`22.2%`)

This indicates best `d` is strongly dependent on horizon/regime assumptions and should be selected per feature family and objective, not globally.

---

## Asset-Level Notes (Run A)

Top best-d assets by event IC-IR:
- `1INCH/USDT`: best `d=0.4`, `ic_ir_event=5.7703`
- `ACM/USDT`: best `d=0.6`, `ic_ir_event=4.6789`
- `ADX/USDT`: best `d=0.6`, `ic_ir_event=3.2036`
- `ADA/USDT`: best `d=0.2`, `ic_ir_event=2.5079`

Weak/negative event IC-IR assets:
- `1000CAT/USDC`: best available `d=0.3`, `ic_ir_event=-0.7593`
- `1000CAT/USDT`: best available `d=0.6`, `ic_ir_event=-0.2636`

Implication: some assets may require exclusion, separate preprocessing, or different targets.

---

## Practical Pros / Cons by d

### d = 0.2
- Pros: strongest average raw event IC in Run A.
- Cons: very high warmup/compute cost; weak in short-horizon Run B.
- Use when: slower trend/context families where memory matters.

### d = 0.3
- Pros: strong long-horizon IC-IR; lower cost than 0.2.
- Cons: degrades under short-horizon setup.
- Use when: mixed trend/MR context families.

### d = 0.4
- Pros: best long-horizon event IC-IR (Run A), balanced memory/cost.
- Cons: not best in short-horizon setup.
- Use when: default robust choice for medium-horizon diagnostics.

### d = 0.5
- Pros: lower warmup/compute; improves in short-horizon setup.
- Cons: weaker in long-horizon event IC-IR than 0.3/0.4.
- Use when: fast momentum/impulse families.

### d = 0.6
- Pros: best short-horizon IC-IR; lowest warmup/compute burden.
- Cons: weakest in long-horizon setup; can be noisier.
- Use when: intraday/fast-reaction features, with validation guards.

---

## Recommended Operating Policy

1. Keep **multi-d** production set (already implemented):
   - trend/slope + MR distance: `d in {0.2, 0.3}`
   - momentum/impulse: `d in {0.4, 0.5}`
   - experimental fast channel: `d=0.6`
2. Prefer `d=0.4` as default diagnostic anchor for longer-horizon event objectives.
3. For short-horizon labels, allow elevated weight for `d in {0.5, 0.6}` families.
4. Gate low-quality assets (negative event IC-IR across all d) from training universe.

---

## Key Takeaway

There is no single globally optimal `d` across assets and horizons.
The strongest strategy is **feature-family-specific d assignment + regular cross-asset diagnostics**.
