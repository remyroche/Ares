# Direct top-40% meta target with separate EV-map combination grid

## Experiment

- Meta is trained only on candidates in the broad base model's raw-score top
  40% within each 4-hour × side query.
- Meta target is direct ordered H12 net outcome (`_rank_target(net_bps)`), not
  a residual against the base map.
- The model uses native LambdaRank with 4-hour × side queries.
- Base and meta raw outputs are mapped independently to side-local,
  prior-resolved expected net bps using monotone PAVA.
- The meta LambdaRank score is passed through a fixed monotonic `tanh` before
  EV mapping because LambdaRank scores are unbounded; no labels or future
  outcomes enter this transform.
- Combination is applied only to the admitted top-40%; all other candidates
  retain the base EV score.
- Specialist context uses the cost-aware +50-bps target.

Artifact: `data_perp/artifacts/top40_direct_meta_ev_grid_20260805_v4/`.

## True global OOS result

All values are net bps/trade after the 100-bps cost.

| Base EV weight | Meta EV weight | Top 1% | Top 5% | Top 10% |
|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | −22.33 | **+11.52** | **−43.33** |
| 0.75 | 0.25 | **+45.00** | −8.03 | −42.98 |
| 0.50 | 0.50 | +17.21 | −51.39 | −50.70 |
| 0.25 | 0.75 | −37.40 | −62.03 | −74.78 |
| 0.00 | 1.00 | −57.88 | −72.30 | −80.05 |

The 75/25 mixture improves top-1 by 67.33 bps versus base-only, but does not
improve top-5 or top-10. Its top-1 advantage is not stable across folds:

| Fold | Base top-1 | 75/25 top-1 |
|---|---:|---:|
| Jul–Aug | −64.05 | −159.62 |
| Sep–Oct | −44.39 | −50.42 |
| Nov partial | +131.00 | +151.04 |

The improvement is therefore largely a November/regime effect.

## OOF-selected side-local mixture

Calibration-selected weights were:

| Fold | Long | Short |
|---|---:|---:|
| Jul–Aug | 100/0 | 75/25 |
| Sep–Oct | 50/50 | 25/75 |
| Nov partial | 100/0 | 75/25 |

When replayed OOS, this selected policy produced:

| Policy | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| Base-only | −22.33 | +11.52 | −43.33 |
| OOF-selected mixture | −31.92 | −31.33 | −41.01 |

The OOF grid selection does not transport: it loses at top-1/top-5 and only
slightly improves top-10.

## Side behaviour

The base EV score remains long-dominated in the global top tails. Adding meta EV
introduces short candidates, but short conversion remains poor. For the 75/25
mixture:

- long: +69.80 / +10.29 / −8.03 bps at top 1/5/10%;
- short: −351.01 / −128.74 / −104.42 bps at top 1/5/10%.

The mixture is therefore not fixing the cross-side mapping problem; it is
changing the side composition of global ranking.

## Query and mapping audit

The direct meta queries are genuinely 4-hour × side:

- Jul–Aug: 1,593 queries per side, median 93 admitted rows/query;
- Sep–Oct: 1,959 queries per side, median 92 rows/query;
- Nov partial: 2,325 queries per side, median 92 rows/query.

Both maps use the strict prior boundary
`label_available_ts < decision_timestamp`.

## Decision

The direct-meta EV-map combination is not an advancement. The only promising
point, 75/25, is unstable by fold and damages broader tails. The base-only
side-local EV map remains the safer control. A useful follow-up would need an
explicit side-comparability/admission constraint before allowing meta EV to
enter global ranking.

