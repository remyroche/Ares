# Strict-R3 six EV/capacity mapper-family ablation — 2026-08-13

## Decision

The strongest untouched-2026 research challenger is **MC1_d2**: frozen score plus contemporaneous model-agreement geometry, fitted as a shallow depth-2 calibration model, plus a causal recent-global shift. It is not automatically canonical: this is its first opened confirmation period and August has already influenced adjacent mapper research.

MC1_d2 is preferred over MC1_d3 and MC3_d2 because it is the only finalist with all 31 observed 2026 weeks positive under the matched frozen-ranking portfolio replay.

## Causal contract

- Long-only strict-prequential ledger: April 2024–July 2026.
- 2025 only: mapper structure/HPO selection.
- 2026: opened once after selecting the top three per family.
- Outcomes enter a mapper only when `policy_label_available_ts <= decision_ts`.
- Ranking is the frozen strict-R3 `final_score`; mappers control admission/capacity only.
- Exact frozen SimplePolicyOptimiser outcome: next-bar entry, H12 maximum, 100 bps cost exactly once.
- Portfolio: canonical global auction, 7x leverage, 10% margin slots.
- Training sufficient statistics use a deterministic day-balanced sample retaining every top-50 row plus 250 broad rows/day. Full populations are scored and replayed.

## 2026 constrained replay

| Arm | Trades | Trades/day | Net bps/trade | Net sum bps | Positive weeks | Worst week | Sortino | Max MTM DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust-21 | 2,761 | 13.22 | +127.48 | +351,981 | 24/26 | -56.4 bps/trade | 0.460 | -65.0% |
| Robust-28 | 2,778 | 13.23 | +118.84 | +330,145 | 23/26 | -83.9 | — | — |
| MC1 depth-2 | 3,855 | 18.19 | **+155.15** | **+598,095** | **31/31** | **+1.3** | **0.755** | **-38.5%** |
| MC1 depth-3 | 3,839 | 18.11 | +157.19 | +603,464 | 29/31 | -13.0 | 0.745 | -39.6% |
| MC3 depth-2 | 3,897 | 18.39 | +152.25 | +593,314 | 29/31 | -7.7 | 0.780 | -38.3% |
| EB affine 28d | 291 | 5.82 active-day rate | +139.87 | +40,701 | 6/6 | +86.8 | — | — |
| HB cell 21d | 3,919 | 18.53 | +67.01 | +262,596 | 28/31 | -48.5 | — | — |
| Rank capacity 21d | 4,550 | 21.47 | +78.73 | +358,238 | 29/31 | -53.4 | — | — |

The astronomical compounded wallets are not credible forecasts; they are mechanical consequences of repeated 7x compounding. Compare EV/trade, weekly downside, Sortino and drawdown instead.

## Monthly net bps/trade

| Month | Robust-21 | MC1 d2 | MC1 d3 | MC3 d2 |
|---|---:|---:|---:|---:|
| Jan | +97.3 | **+164.2** | +157.6 | +161.9 |
| Feb | +147.2 | **+198.5** | +190.4 | +196.7 |
| Mar | +156.9 | +177.2 | **+203.9** | +178.9 |
| Apr | +197.4 | +217.6 | **+230.4** | +213.7 |
| May | +80.7 | **+105.7** | +100.9 | +100.6 |
| Jun | no trades | **+28.5** | +27.9 | +27.7 |
| Jul | +73.8 | **+155.5** | +143.8 | +148.2 |

MC1_d2 removes the Robust-21 39-day maximum drought and produces at least one accepted trade every day/week in January–July. It does not foresee the February 5 shock, but resumes immediately; during the July shock cluster it avoids the large losses selected by Robust-21 and retains positive subsequent-day EV.

## Family conclusions

- **Agreement calibration (MC): advances.** The signal is stable in 2025 and 2026. Score plus base/consensus/correctness agreement is sufficient; support/OOD additions (MC3) do not improve portability over MC1.
- **Hierarchical Bayes: useful but inferior.** It eliminates drought and is stable, but admits too deeply and halves EV/trade versus MC1.
- **Rank capacity: capacity/recovery diagnostic, not winner.** It trades every day and maximizes count, but marginal quality degrades materially.
- **Empirical Bayes: accurate when active but excessive drought.** The affine 28d arm has high EV but only three active months and 291 constrained trades.
- **Change point: rejected.** Detection/reset combinations remain discontinuous and include a severely negative February.
- **Latent state: rejected.** The scalar opportunity state over-admits; the structural score curve plus state formulation was not sufficient.
- **Robust-21 remains the production control.** MC1_d2 is the research challenger pending later untouched validation.

## Reproducibility

- Runner: `scripts/run_strict_r3_six_mapper_families.py`
- Immutable output: `data_perp/artifacts/strict_r3_six_mapper_families_long_2025_2026_20260813_v4`
- Corrected frozen-rank files end in `_frozenrank.parquet`.
- `portfolio_2026_summary.parquet` without that suffix is superseded because it let mapped EV alter auction order.
