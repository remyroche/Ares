# Long family/path residual handoff

## Current frozen contract

The production research contract is the cross-fold semantic family contract
`long_structural_family_semantic_020_top64_strict_20260808_v1`:

- Jaccard similarity threshold: 0.20; top 64 semantic families retained.
- No nearest-medoid fallback; all selected assignments satisfy the threshold.
- 3,637 selected rule instances across the three chronological folds.
- Mean held-out absolute p-clear contribution mass represented: **88.40%**;
  83.79% of held-out rows have at least 80% represented mass.
- Assignment-quality-weighted mass: 33.87%; low-confidence represented mass:
  54.53%.

The last two figures matter: the 80% coverage gate is met, but much of that
coverage is low-similarity family assignment. Coverage and semantic confidence
must therefore remain separate trust inputs.

## Residual authority-size ablation

Authority selection is fold-local and uses only positive meta-train rank IC.
All 64 families remain available for coverage/trust diagnostics; only the
authority subset is passed to the residual correction features.

| authority | H top 0.5% | H top 1% | H top 5% | H top 10% | G top 5% |
|---|---:|---:|---:|---:|---:|
| all 64 | **−22.24** | **−35.58** | **−55.59** | **−65.71** | **−54.99** |
| top 8 | −43.50 | −44.02 | −56.87 | −67.09 | −56.82 |
| top 12 | −24.77 | −38.51 | −55.74 | −67.27 | −56.30 |
| top 16 | −49.78 | −49.46 | −56.37 | −66.56 | −56.92 |

No restricted authority subset improves the all-family H arm. Keep all 64
families for the current residual input contract; use fold-local authority
selection only as a diagnostic.

A predeclared stability filter requiring positive signed family IC in both
meta-train and pre-test calibration was also tested at K=8. It produced H
top-5 -55.96 bps/trade and monthly mean -66.51 bps/trade, versus -55.59 and
-63.20 for the raw all-family H control. It is not promoted; stability should
be exposed as a continuous trust feature rather than a hard filter.

## Leaf value/contribution weighting

Three causal weighting modes were replayed against the same all-family contract:

- `value`: signed path contribution × bounded absolute frozen leaf value;
- `contribution`: signed path contribution × bounded emitted contribution
  strength;
- `value_x_contribution`: geometric combination of the two factors.

The factors are fit from the frozen training leaf catalogs only and capped at
`[0.25, 4]`. No realised outcome is used.

| weighting | H top 0.5% | H top 1% | H top 5% | H top 10% |
|---|---:|---:|---:|---:|
| raw control | **−22.24** | **−35.58** | **−55.59** | **−65.71** |
| value | −53.46 | −47.47 | −56.39 | −67.35 |
| contribution | −38.06 | −43.51 | −58.24 | −68.51 |
| value × contribution | −48.73 | −50.46 | −61.22 | −67.71 |

The weighting variants increase the mechanically defined contribution mass to
89.77–90.23%, but do not improve economics. The raw signed contribution
contract remains the winner.

## Economic interpretation

The family layer now meets the requested mass-coverage requirement, but it does
not solve the conversion problem:

1. The Cap-120 base score remains the strongest broad ranking component in this
   replay.
2. Family-state, MLP, reliability, and leaf-weighting corrections can improve
   the very narrow top 0.5–1% in some folds, but are negative at top 5–10%.
3. The persistent negative top-5 result is not explained by missing family
   coverage alone. The dominant remaining issue is that family correctness is
   weakly portable and low-confidence mass is large.
4. Additional leaf-strength weighting should not be promoted. It concentrates
   the same base evidence rather than adding independent information.

## Recommended next gate

Freeze the raw 64-family contribution representation and stop expanding leaf
weighting variants. The next experiment should target the semantic-confidence
gap: improve stable cross-fold family identity, or add causal regime/trust
features that tell the residual layer when a low-confidence family assignment
should be ignored. Any new correction must beat the raw all-family H arm on
pooled top-5 **and** avoid a worse worst-month result before promotion.

## Reproducible artifacts

- [strict family contract](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_structural_family_semantic_020_top64_strict_20260808_v1)
- [raw all-family replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_conditional_correctness_semantic020_top64_strict_20260808_v1)
- [value-weighted replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_value_weighted_semantic020_top64_20260808_v2)
- [contribution-weighted replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_contribution_weighted_semantic020_top64_20260808_v1)
- [combined replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_leaf_value_x_contribution_semantic020_top64_20260808_v1)
- [replay implementation](/Users/remyroche/Documents/Ares/scripts/run_long_family_conditional_correctness.py)
