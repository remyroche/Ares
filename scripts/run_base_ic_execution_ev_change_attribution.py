#!/usr/bin/env python3
"""Attribute month-to-month base-rank versus execution-EV divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_top(frame: pd.DataFrame, score_column: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(float(fraction) * len(frame))))
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order[:count]].copy()


def _corr(x: pd.Series, y: pd.Series) -> float:
    result = spearmanr(
        pd.to_numeric(x, errors="raise"),
        pd.to_numeric(y, errors="raise"),
        nan_policy="omit",
    ).statistic
    return float(result)


def bridge_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "native_alpha_soft_24h": "__first_touch_target_soft__",
        "exact_mfe_12h": "execution_mfe_return_12h",
        "exact_gross_12h": "execution_gross_ev_12h",
        "exact_net_12h": "execution_net_ev_12h",
        "opportunity_0bps": "opportunity_gross_above_cost_0bps",
        "trailing_exit": "exit_is_trailing",
        "timeout_exit": "exit_is_timeout",
        "full_stop_exit": "exit_is_full_stop",
    }
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled_sides", frame)]
    scopes.extend(
        (side, local) for side, local in frame.groupby("side_name", sort=True)
    )
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        month_scopes = [("pooled_sides", month_rows)]
        month_scopes.extend(
            (side, local)
            for side, local in month_rows.groupby("side_name", sort=True)
        )
        for scope, local in month_scopes:
            for target_name, target_column in targets.items():
                if target_column not in local:
                    continue
                rows.append(
                    {
                        "candidate_month": month,
                        "scope": scope,
                        "target": target_name,
                        "rows": int(len(local)),
                        "spearman": _corr(local["score_raw"], local[target_column]),
                    }
                )
    return pd.DataFrame(rows)


def selected_month_components(
    frame: pd.DataFrame,
    *,
    fraction: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    selected_by_month: dict[str, pd.DataFrame] = {}
    for month, local in frame.groupby("candidate_month", sort=True):
        selected = _stable_top(local, "score_raw", fraction)
        selected_by_month[str(month)] = selected
        opportunity = selected["opportunity_gross_above_cost_0bps"].astype(bool)
        gross = selected["execution_gross_ev_12h"].astype(float)
        cost = selected["execution_cost_return"].astype(float)
        net = selected["execution_net_ev_12h"].astype(float)
        row: dict[str, Any] = {
            "candidate_month": month,
            "candidate_rows": int(len(local)),
            "selected_rows": int(len(selected)),
            "mean_gross_bps": float(10_000.0 * gross.mean()),
            "mean_cost_bps": float(10_000.0 * cost.mean()),
            "mean_net_bps": float(10_000.0 * net.mean()),
            "opportunity_rate": float(opportunity.mean()),
            "opportunity_conditional_gross_bps": float(
                10_000.0 * gross.loc[opportunity].mean()
            ),
            "no_opportunity_conditional_gross_bps": float(
                10_000.0 * gross.loc[~opportunity].mean()
            ),
        }
        for side in ("long", "short"):
            mask = selected["side_name"].eq(side)
            row[f"{side}_share"] = float(mask.mean())
            row[f"{side}_conditional_gross_bps"] = float(
                10_000.0 * gross.loc[mask].mean()
            )
        for exit_class in EXIT_CLASSES:
            mask = selected["execution_exit_class"].eq(exit_class)
            row[f"{exit_class}_share"] = float(mask.mean())
            row[f"{exit_class}_conditional_gross_bps"] = float(
                10_000.0 * gross.loc[mask].mean()
            )
            row[f"{exit_class}_conditional_net_bps"] = float(
                10_000.0 * net.loc[mask].mean()
            )
        summaries.append(row)
    return pd.DataFrame(summaries), selected_by_month


def _two_state_shapley(
    probability_a: np.ndarray,
    value_a: np.ndarray,
    probability_b: np.ndarray,
    value_b: np.ndarray,
) -> tuple[float, float]:
    probability_effect = 0.5 * (
        np.dot(probability_b - probability_a, value_a)
        + np.dot(probability_b - probability_a, value_b)
    )
    value_effect = 0.5 * (
        np.dot(probability_a, value_b - value_a)
        + np.dot(probability_b, value_b - value_a)
    )
    return float(probability_effect), float(value_effect)


def change_attribution(summary: pd.DataFrame) -> pd.DataFrame:
    by_month = summary.set_index("candidate_month")
    months = list(by_month.index)
    rows: list[dict[str, Any]] = []
    for month_a, month_b in zip(months, months[1:]):
        a = by_month.loc[month_a]
        b = by_month.loc[month_b]
        cost_effect = -(
            float(b["mean_cost_bps"]) - float(a["mean_cost_bps"])
        )
        actual_delta = float(b["mean_net_bps"]) - float(a["mean_net_bps"])
        opportunity_probability_a = np.array(
            [a["opportunity_rate"], 1.0 - a["opportunity_rate"]], dtype=float
        )
        opportunity_probability_b = np.array(
            [b["opportunity_rate"], 1.0 - b["opportunity_rate"]], dtype=float
        )
        opportunity_value_a = np.array(
            [
                a["opportunity_conditional_gross_bps"],
                a["no_opportunity_conditional_gross_bps"],
            ],
            dtype=float,
        )
        opportunity_value_b = np.array(
            [
                b["opportunity_conditional_gross_bps"],
                b["no_opportunity_conditional_gross_bps"],
            ],
            dtype=float,
        )
        prevalence, conditional_payoff = _two_state_shapley(
            opportunity_probability_a,
            opportunity_value_a,
            opportunity_probability_b,
            opportunity_value_b,
        )
        exit_probability_a = np.array(
            [a[f"{name}_share"] for name in EXIT_CLASSES], dtype=float
        )
        exit_probability_b = np.array(
            [b[f"{name}_share"] for name in EXIT_CLASSES], dtype=float
        )
        exit_value_a = np.array(
            [a[f"{name}_conditional_gross_bps"] for name in EXIT_CLASSES],
            dtype=float,
        )
        exit_value_b = np.array(
            [b[f"{name}_conditional_gross_bps"] for name in EXIT_CLASSES],
            dtype=float,
        )
        exit_mix, exit_payoff = _two_state_shapley(
            exit_probability_a, exit_value_a, exit_probability_b, exit_value_b
        )
        side_probability_a = np.array(
            [a["long_share"], a["short_share"]], dtype=float
        )
        side_probability_b = np.array(
            [b["long_share"], b["short_share"]], dtype=float
        )
        side_value_a = np.array(
            [a["long_conditional_gross_bps"], a["short_conditional_gross_bps"]],
            dtype=float,
        )
        side_value_b = np.array(
            [b["long_conditional_gross_bps"], b["short_conditional_gross_bps"]],
            dtype=float,
        )
        side_mix, within_side_payoff = _two_state_shapley(
            side_probability_a, side_value_a, side_probability_b, side_value_b
        )
        lenses = {
            "opportunity": (
                "opportunity_prevalence",
                prevalence,
                "conditional_opportunity_payoff",
                conditional_payoff,
            ),
            "exit": ("exit_mix", exit_mix, "conditional_exit_payoff", exit_payoff),
            "side_book": (
                "side_mix",
                side_mix,
                "within_side_payoff",
                within_side_payoff,
            ),
        }
        for lens, (
            first_name,
            first_value,
            second_name,
            second_value,
        ) in lenses.items():
            reconstructed = first_value + second_value + cost_effect
            rows.append(
                {
                    "from_month": month_a,
                    "to_month": month_b,
                    "lens": lens,
                    "actual_net_delta_bps": actual_delta,
                    first_name: first_value,
                    second_name: second_value,
                    "cost_effect_bps": cost_effect,
                    "reconstructed_delta_bps": reconstructed,
                    "reconciliation_error_bps": reconstructed - actual_delta,
                }
            )
    return pd.DataFrame(rows)


def rank_conversion_counterfactual(
    frame: pd.DataFrame,
    selected_by_month: Mapping[str, pd.DataFrame],
    *,
    bins: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["month_side_rank_pct"] = work.groupby(
        ["candidate_month", "side_name"], sort=False
    )["score_raw"].rank(method="first", pct=True)
    work["rank_bin"] = np.minimum(
        np.floor(work["month_side_rank_pct"] * int(bins)).astype(int),
        int(bins) - 1,
    )
    cell = (
        work.groupby(["candidate_month", "side_name", "rank_bin"], sort=True)
        ["execution_net_ev_12h"]
        .mean()
        .mul(10_000.0)
        .rename("cell_net_bps")
        .reset_index()
    )
    weights: dict[str, pd.Series] = {}
    values: dict[str, pd.Series] = {}
    for month, selected in selected_by_month.items():
        keys = work.loc[
            work["candidate_id"].isin(selected["candidate_id"]),
            ["side_name", "rank_bin"],
        ]
        weights[month] = (
            keys.value_counts(normalize=True)
            .rename("weight")
            .sort_index()
        )
        values[month] = (
            cell.loc[cell["candidate_month"].eq(month)]
            .set_index(["side_name", "rank_bin"])["cell_net_bps"]
            .sort_index()
        )
    months = sorted(weights)
    rows: list[dict[str, Any]] = []
    for month_a, month_b in zip(months, months[1:]):
        index = weights[month_a].index.union(weights[month_b].index)
        wa = weights[month_a].reindex(index, fill_value=0.0).to_numpy(float)
        wb = weights[month_b].reindex(index, fill_value=0.0).to_numpy(float)
        ma = values[month_a].reindex(index).to_numpy(float)
        mb = values[month_b].reindex(index).to_numpy(float)
        finite = np.isfinite(ma) & np.isfinite(mb)
        if not finite.all():
            dropped_weight = max(
                float(wa[~finite].sum()), float(wb[~finite].sum())
            )
            wa = wa[finite]
            wb = wb[finite]
            ma = ma[finite]
            mb = mb[finite]
            wa /= wa.sum()
            wb /= wb.sum()
        else:
            dropped_weight = 0.0
        ordering, conversion = _two_state_shapley(wa, ma, wb, mb)
        modeled_a = float(np.dot(wa, ma))
        modeled_b = float(np.dot(wb, mb))
        rows.append(
            {
                "from_month": month_a,
                "to_month": month_b,
                "rank_bins_per_side": int(bins),
                "from_ordering_from_conversion_bps": modeled_a,
                "from_ordering_to_conversion_bps": float(np.dot(wa, mb)),
                "to_ordering_from_conversion_bps": float(np.dot(wb, ma)),
                "to_ordering_to_conversion_bps": modeled_b,
                "ordering_composition_effect_bps": ordering,
                "rank_to_economics_conversion_effect_bps": conversion,
                "modeled_delta_bps": modeled_b - modeled_a,
                "maximum_dropped_weight": dropped_weight,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    frame = pd.read_parquet(args.mapped_candidates)
    required = {
        "candidate_id",
        "candidate_month",
        "side_name",
        "score_raw",
        "__first_touch_target_soft__",
        "execution_exit_class",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "opportunity_gross_above_cost_0bps",
        "exit_is_trailing",
        "exit_is_timeout",
        "exit_is_full_stop",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"attribution input missing fields: {missing}")
    if frame["source_family"].nunique() != 1:
        raise ValueError("attribution input mixes source families")
    bridge = bridge_metrics(frame)
    summary, selected = selected_month_components(
        frame, fraction=args.top_k_fraction
    )
    attribution = change_attribution(summary)
    counterfactual = rank_conversion_counterfactual(
        frame, selected, bins=args.rank_bins
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "bridge": args.output_dir / "score_target_bridge.csv",
        "selected": args.output_dir / "selected_month_components.csv",
        "attribution": args.output_dir / "month_change_attribution.csv",
        "counterfactual": args.output_dir / "rank_conversion_counterfactual.csv",
    }
    bridge.to_csv(outputs["bridge"], index=False)
    summary.to_csv(outputs["selected"], index=False)
    attribution.to_csv(outputs["attribution"], index=False)
    counterfactual.to_csv(outputs["counterfactual"], index=False)
    manifest = {
        "schema": "base_ic_execution_ev_change_attribution_v1",
        "source": {
            "path": str(args.mapped_candidates),
            "sha256": _sha256(args.mapped_candidates),
        },
        "source_family": str(frame["source_family"].iloc[0]),
        "selection": {
            "score": "score_raw",
            "top_k_fraction": float(args.top_k_fraction),
            "scope": "one pooled global selection within month, never per timestamp",
            "tie_break": "candidate_id ascending",
        },
        "interpretation": {
            "opportunity_exit_and_side_lenses": (
                "three independent exact Shapley reconciliations of the same "
                "gross-minus-cost month delta; their effects are not additive across lenses"
            ),
            "rank_counterfactual": (
                "coarse side x within-month score-rank-bin attribution with "
                "ordering/conversion interaction split symmetrically"
            ),
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapped-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--rank-bins", type=int, default=100)
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
