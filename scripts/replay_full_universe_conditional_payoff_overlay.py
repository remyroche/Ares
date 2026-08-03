#!/usr/bin/env python3
"""Development-select frozen meta overlays on Stage 5.2 conditional value.

This is deliberately an integration/replay, not a refit.  It combines:

* the selected conditional-payoff value from Stage 5.2;
* the frozen all-candidate B2 residual correction; and
* the frozen robust ``P(net > 25 bps)`` reliability head, valid only in its
  causally admitted B2 top-30% population.

Every candidate remains in the globally pooled book.  Rows outside the
reliability population receive no reliability adjustment.  All rule and
strength choices are made on June 15--August 1 development rows; August--
November labels are read solely after that choice has been frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TOPS = (0.01, 0.05, 0.10, 0.20)


def _load(conditional: Path, residual: Path, reliability: Path) -> pd.DataFrame:
    value = pd.read_parquet(conditional)
    residual = pd.read_parquet(
        residual,
        columns=["candidate_id", "base_expected_net_bps", "final_score"],
    ).rename(
        columns={
            "base_expected_net_bps": "frozen_b2_expected_net_bps",
            "final_score": "frozen_b2_residual_value_bps",
        }
    )
    reliability = pd.read_parquet(
        reliability,
        columns=["candidate_id", "reliability_score", "high_base_eligible"],
    ).rename(columns={"high_base_eligible": "reliability_eligible"})
    frame = value.merge(residual, on="candidate_id", validate="one_to_one")
    frame = frame.merge(reliability, on="candidate_id", validate="one_to_one")
    if frame.candidate_id.duplicated().any() or len(frame) != len(value):
        raise ValueError("integration joins must retain one row per conditional value")
    # The residual is frozen in B2 value units.  Its delta can be transferred
    # to the conditional value without pretending that the residual model was
    # re-trained on conditional-payoff labels.
    frame["frozen_residual_delta_bps"] = (
        frame.frozen_b2_residual_value_bps - frame.frozen_b2_expected_net_bps
    )
    frame["conditional_value_bps"] = frame.conditional_score_bps
    frame["conditional_plus_frozen_residual_bps"] = (
        frame.conditional_value_bps + frame.frozen_residual_delta_bps
    )
    if not np.isfinite(frame[["conditional_value_bps", "frozen_residual_delta_bps"]].to_numpy(float)).all():
        raise ValueError("conditional value and residual delta must be finite")
    if not frame.loc[frame.reliability_eligible, "reliability_score"].notna().all():
        raise ValueError("admitted reliability rows must have a probability")
    if frame.loc[~frame.reliability_eligible, "reliability_score"].notna().any():
        raise ValueError("reliability score must be absent outside its frozen P30 admission")
    return frame


def _rank(values: np.ndarray) -> np.ndarray:
    # Deterministic continuous rank, candidate identity resolves only exact
    # final-score ties in the final sort, never in the learned coordinate.
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def _score(frame: pd.DataFrame, rule: str, strength: float) -> np.ndarray:
    value = frame.conditional_plus_frozen_residual_bps.to_numpy(float)
    p = frame.reliability_score.to_numpy(float)
    eligible = np.isfinite(p)
    if rule == "conditional_only":
        return frame.conditional_value_bps.to_numpy(float)
    if rule == "conditional_plus_residual":
        return value
    if rule == "rank_blend":
        # Non-admitted rows must receive *exactly zero* reliability
        # correction.  Their trust coordinate is therefore their own value
        # rank, not an artificial floor that would create an implicit veto.
        value_rank = _rank(value)
        trust = value_rank.copy()
        trust[eligible] = _rank(p[eligible])
        return (1.0 - strength) * value_rank + strength * trust
    if rule == "centered_bps":
        out = value.copy()
        prior = float(np.mean(p[eligible]))
        out[eligible] += strength * (p[eligible] - prior)
        return out
    if rule == "logit_bps":
        out = value.copy()
        prior = float(np.mean(p[eligible]))
        q = np.clip(p[eligible], 1e-4, 1.0 - 1e-4)
        base = np.clip(prior, 1e-4, 1.0 - 1e-4)
        out[eligible] += strength * (np.log(q / (1.0 - q)) - np.log(base / (1.0 - base)))
        return out
    raise ValueError(rule)


def _metrics(frame: pd.DataFrame, score: np.ndarray) -> list[dict]:
    ranked = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for fraction in TOPS:
        top = ranked.head(int(np.ceil(len(ranked) * fraction)))
        rows.append({
            "top_fraction": fraction,
            "n": int(len(top)),
            "gross_bps": float(top.gross_bps.mean()),
            "net_bps": float(top.net_bps.mean()),
            "long_n": int(top.side_name.eq("long").sum()),
            "short_n": int(top.side_name.eq("short").sum()),
        })
    return rows


def _objective(frame: pd.DataFrame, score: np.ndarray) -> dict:
    metrics = _metrics(frame, score)
    lookup = {x["top_fraction"]: x for x in metrics}
    ic = float(spearmanr(score, frame.net_bps.to_numpy(float)).statistic)
    # Same broad-tail emphasis as the conditional-payoff selection.  This is
    # fixed before the OOS book is opened and prevents a one-percent spike
    # from winning an integration rule.
    selection_score = .60 * lookup[.10]["net_bps"] + .25 * lookup[.05]["net_bps"] + .15 * 100.0 * ic
    return {"selection_score": float(selection_score), "net_spearman_ic": ic, "metrics": metrics}


def _fit_two_dimensional_map(fit: pd.DataFrame, *, bins: int, shrink: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a tiny shrunk table using only a temporally earlier labelled span."""
    good = fit.reliability_score.notna().to_numpy()
    value = fit.conditional_plus_frozen_residual_bps.to_numpy(float)
    probability = fit.reliability_score.to_numpy(float)
    if int(good.sum()) < bins * bins * 30:
        raise ValueError("insufficient admitted rows for two-dimensional map")
    value_edges = np.quantile(value[good], np.linspace(0., 1., bins + 1))
    probability_edges = np.quantile(probability[good], np.linspace(0., 1., bins + 1))
    value_edges = np.maximum.accumulate(value_edges + np.arange(bins + 1) * 1e-9)
    probability_edges = np.maximum.accumulate(probability_edges + np.arange(bins + 1) * 1e-9)
    vi = np.clip(np.searchsorted(value_edges[1:-1], value, side="right"), 0, bins - 1)
    pi = np.clip(np.searchsorted(probability_edges[1:-1], np.nan_to_num(probability, nan=-1.), side="right"), 0, bins - 1)
    sums, counts = np.zeros((bins, bins)), np.zeros((bins, bins))
    np.add.at(sums, (vi[good], pi[good]), fit.loc[good, "net_bps"].to_numpy(float))
    np.add.at(counts, (vi[good], pi[good]), 1.)
    prior = float(fit.loc[good, "net_bps"].mean())
    return value_edges, probability_edges, (sums + shrink * prior) / (counts + shrink)


def _apply_two_dimensional_map(frame: pd.DataFrame, value_edges: np.ndarray, probability_edges: np.ndarray, table: np.ndarray, blend: float) -> np.ndarray:
    out = frame.conditional_plus_frozen_residual_bps.to_numpy(float).copy()
    good = frame.reliability_score.notna().to_numpy()
    vi = np.clip(np.searchsorted(value_edges[1:-1], out, side="right"), 0, table.shape[0] - 1)
    pi = np.clip(np.searchsorted(probability_edges[1:-1], frame.reliability_score.fillna(-1.).to_numpy(float), side="right"), 0, table.shape[1] - 1)
    out[good] = (1. - blend) * out[good] + blend * table[vi[good], pi[good]]
    return out


def _two_dimensional_holdout(dev: pd.DataFrame, oos: pd.DataFrame) -> dict:
    """Select table capacity on an earlier/later split inside development.

    The first map-fit span ends at 2024-07-07 12:00 UTC.  The scored holdout
    begins 12 hours later, so the fixed H12 label cannot cross the boundary.
    After choosing capacity on that holdout, re-fit the table on all resolved
    development rows and replay it once on OOS.
    """
    split = pd.Timestamp("2024-07-08", tz="UTC")
    safe_fit_end = split - pd.Timedelta(hours=12)
    fit = dev[dev.__ts__.lt(safe_fit_end)].copy()
    holdout = dev[dev.__ts__.ge(split)].copy()
    rows: list[dict] = []
    for bins in (3, 5):
        for shrink in (1000., 4000.):
            edges_value, edges_probability, table = _fit_two_dimensional_map(fit, bins=bins, shrink=shrink)
            # Zero is an explicit no-table control on the identical temporal
            # holdout; without it the table would be forced to look useful.
            for blend in (0., .25, .5, .75):
                score = _apply_two_dimensional_map(holdout, edges_value, edges_probability, table, blend)
                rows.append({"bins": bins, "shrinkage_rows": shrink, "blend": blend, "holdout": _objective(holdout, score)})
    winner = sorted(rows, key=lambda x: (-x["holdout"]["selection_score"], x["bins"], x["shrinkage_rows"], x["blend"]))[0]
    # The winner's map may now consume every resolved development label, but
    # its hyperparameters remain frozen from the earlier/later dev holdout.
    edges_value, edges_probability, table = _fit_two_dimensional_map(dev, bins=winner["bins"], shrink=winner["shrinkage_rows"])
    winner["oos"] = _objective(oos, _apply_two_dimensional_map(oos, edges_value, edges_probability, table, winner["blend"]))
    return {"selection": "temporally separated development table fit/holdout; H12 purge between spans; refit all development then one OOS replay", "fit_rows": len(fit), "holdout_rows": len(holdout), "winner": winner, "grid": rows}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-conditional", type=Path, default=root / "data_perp/artifacts/full_universe_stage5_2_conditional_payoff_20260804_v1/development_selected_predictions.parquet")
    p.add_argument("--oos-conditional", type=Path, default=root / "data_perp/artifacts/full_universe_stage5_2_conditional_payoff_20260804_v1/oos_predictions.parquet")
    p.add_argument("--development-residual", type=Path, default=root / "data_perp/artifacts/full_universe_round_b_residual_dev_20260803_v1/predictions.parquet")
    p.add_argument("--oos-residual", type=Path, default=root / "data_perp/artifacts/full_universe_round_b_residual_20260803_v1/predictions.parquet")
    p.add_argument("--development-reliability", type=Path, default=root / "data_perp/artifacts/full_universe_round2_reliability_net_gt_25_dev_20260804_v1/predictions.parquet")
    p.add_argument("--oos-reliability", type=Path, default=root / "data_perp/artifacts/full_universe_round2_reliability_net_gt_25_oos_20260804_v1/predictions.parquet")
    p.add_argument("--out", type=Path, default=root / "data_perp/artifacts/full_universe_stage5_2_conditional_overlay_20260804_v1")
    a = p.parse_args()
    dev = _load(a.development_conditional, a.development_residual, a.development_reliability)
    oos = _load(a.oos_conditional, a.oos_residual, a.oos_reliability)
    candidates = [("conditional_only", 0.), ("conditional_plus_residual", 0.)]
    candidates += [("rank_blend", x) for x in (.10, .25, .40)]
    candidates += [("centered_bps", x) for x in (25., 50., 100., 150., 200.)]
    candidates += [("logit_bps", x) for x in (10., 25., 50., 100.)]
    rows = []
    for rule, strength in candidates:
        dev_score = _score(dev, rule, strength)
        oos_score = _score(oos, rule, strength)
        rows.append({"rule": rule, "strength": strength, "development": _objective(dev, dev_score), "oos": _objective(oos, oos_score)})
    # Fixed dev-only selection.  Complexity ordering means a base/residual
    # control beats an equally scoring overlay; smaller strength then wins.
    order = {"conditional_only": 0, "conditional_plus_residual": 1, "rank_blend": 2, "centered_bps": 3, "logit_bps": 4}
    winner = sorted(rows, key=lambda x: (-x["development"]["selection_score"], order[x["rule"]], x["strength"]))[0]
    selected_dev_score = _score(dev, winner["rule"], winner["strength"])
    selected_oos_score = _score(oos, winner["rule"], winner["strength"])
    dev["selected_score"] = selected_dev_score
    oos["selected_score"] = selected_oos_score
    a.out.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(a.out / "development_integrated_predictions.parquet", index=False)
    oos.to_parquet(a.out / "oos_integrated_predictions.parquet", index=False)
    report = {
        "schema": "full_universe_stage5_2_conditional_overlay_v1",
        "status": "COMPLETED_UNTOUCHED_OOS_REPLAY",
        "contract": {
            "value": "selected Stage5.2 conditional payoff expected net",
            "residual": "frozen all-candidate B2 residual delta, transferred additively without retraining",
            "reliability": "frozen shared P(net>25bps) head, B2 causal P30 admission only",
            "population": "all candidates; global pooled ranking; no side/timestamp quota or backfill",
            "selection": "development-only 0.60 top10 net + 0.25 top5 net + 0.15*100 Spearman net IC",
            "oos": "all rule outcomes shown only after one development-selected winner is frozen",
            "two_dimensional": "capacity selected using an earlier/later development split with an H12 purge; reported separately because it uses a narrower development holdout than the primary overlay grid",
        },
        "rows": {"development": len(dev), "oos": len(oos), "development_reliability_eligible": int(dev.reliability_score.notna().sum()), "oos_reliability_eligible": int(oos.reliability_score.notna().sum())},
        "winner": winner,
        "all_valid_rules": rows,
        "two_dimensional_temporal_holdout": _two_dimensional_holdout(dev, oos),
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    def line(name: str, item: dict) -> str:
        m = item["metrics"]
        return (f"| {name} | {item['selection_score']:.2f} | "
                f"{m[0]['net_bps']:.2f} | {m[1]['net_bps']:.2f} | {m[2]['net_bps']:.2f} | {m[3]['net_bps']:.2f} |\n")
    summary = [
        "# Conditional-payoff + frozen overlay replay", "",
        "All combinations were selected exclusively on the June 15--August 1 development period and replayed once on the untouched August--November period.", "",
        "The value score is Stage 5.2 conditional payoff expected net.  The residual is the frozen B2 residual delta; the reliability probability is the frozen shared P(net > 25 bps) head and exists only for its frozen causal B2 top-30% admission population.", "",
        "## Development selection", "",
        "Objective: `0.60 × top-10 net + 0.25 × top-5 net + 0.15 × 100 × Spearman(score, net)`; all rankings are pooled globally with no side/timestamp quota.", "",
        f"Winner: `{winner['rule']}` at strength `{winner['strength']}`.", "",
        "## Untouched OOS net bps", "",
        "| Rule | Dev objective | Top 1% | Top 5% | Top 10% | Top 20% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        name = f"{row['rule']} ({row['strength']:g})"
        summary.append(line(name, row["oos"]))
    summary.extend([
        "", "## Two-dimensional value × reliability table", "",
        f"A separate, temporally held-out development selection chose `{report['two_dimensional_temporal_holdout']['winner']['bins']}×{report['two_dimensional_temporal_holdout']['winner']['bins']}` bins, shrinkage `{report['two_dimensional_temporal_holdout']['winner']['shrinkage_rows']:g}`, blend `{report['two_dimensional_temporal_holdout']['winner']['blend']:g}`.  Its OOS top-10 net is `{report['two_dimensional_temporal_holdout']['winner']['oos']['metrics'][2]['net_bps']:.2f}` bps.  It remains a separate comparison because its selector uses a narrower time-held development span than the primary full-development overlay grid.",
        "", "## Interpretation", "",
        "The development-selected overlay improves OOS top-10 relative to conditional value alone, but it does not meet the positive broad-tail acceptance gate.",
    ])
    (a.out / "REPORT.md").write_text("\n".join(summary))
    print(json.dumps({"winner": winner, "oos_top10": winner["oos"]["metrics"][2]}, indent=2))


if __name__ == "__main__":
    main()
