#!/usr/bin/env python3
"""Diagnose improving base IC versus deteriorating global-tail execution EV.

This is a read-only analysis over the immutable full canonical panel.  It
separates full-population ordering from tail ordering, opportunity prevalence,
conditional payoff, score monotonicity, score compression and book
composition.  Every selection is one pooled global monthly top-k with
candidate-ID tie breaking; timestamp/side groupings are diagnostics only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_base_ic_execution_ev_change_attribution import (
    rank_conversion_counterfactual,
    selected_month_components,
)


PANEL_ROOT = (
    ROOT
    / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
)
OUT = (
    ROOT
    / "data_perp/artifacts/canonical_base_ic_ev_tail_diagnostic_20260729_v1"
)
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
TARGETS = {
    "native_alpha_soft_24h": "__first_touch_target_soft__",
    "exact_mfe_12h": "execution_mfe_return_12h",
    "exact_gross_12h": "execution_gross_ev_12h",
    "exact_net_12h": "execution_net_ev_12h",
    "opportunity_0bps": "opportunity_gross_above_cost_0bps",
    "opportunity_25bps": "opportunity_gross_above_cost_25bps",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _corr(left: pd.Series, right: pd.Series) -> float:
    if (
        pd.Series(left).nunique(dropna=True) < 2
        or pd.Series(right).nunique(dropna=True) < 2
    ):
        return np.nan
    value = spearmanr(
        pd.to_numeric(left, errors="raise"),
        pd.to_numeric(right, errors="raise"),
        nan_policy="omit",
    ).statistic
    return float(value) if np.isfinite(value) else np.nan


def stable_top(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    score = frame["score_raw"].to_numpy(float)
    order = np.lexsort(
        (frame["candidate_id"].astype(str).to_numpy(), -score)
    )
    return frame.iloc[order[:count]].copy()


def _hhi(values: pd.Series) -> float:
    shares = values.value_counts(normalize=True).to_numpy(float)
    return float(np.square(shares).sum())


def full_and_tail_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        overall_0 = float(month_rows["opportunity_gross_above_cost_0bps"].mean())
        overall_25 = float(month_rows["opportunity_gross_above_cost_25bps"].mean())
        for scope, local in [
            ("pooled_global", month_rows),
            *[
                (f"side_{side}", side_rows)
                for side, side_rows in month_rows.groupby("side_name", sort=True)
            ],
        ]:
            base = {
                "candidate_month": str(month),
                "scope": scope,
                "fraction": 1.0,
                "rows": int(len(local)),
                "mean_gross_bps": float(local.execution_gross_ev_12h.mean() * 1e4),
                "mean_cost_bps": float(local.execution_cost_return.mean() * 1e4),
                "mean_net_bps": float(local.execution_net_ev_12h.mean() * 1e4),
                "opportunity_0bps_precision": float(
                    local.opportunity_gross_above_cost_0bps.mean()
                ),
                "opportunity_25bps_precision": float(
                    local.opportunity_gross_above_cost_25bps.mean()
                ),
                "opportunity_0bps_lift_vs_month": float(
                    local.opportunity_gross_above_cost_0bps.mean() / overall_0
                ),
                "opportunity_25bps_lift_vs_month": float(
                    local.opportunity_gross_above_cost_25bps.mean() / overall_25
                ),
                "long_share": float(local.side_name.eq("long").mean()),
                "timestamp_hhi": _hhi(local["__ts__"]),
                "asset_hhi": _hhi(local["__symbol__"]),
                "mean_candidate_group_size": float(
                    local["base_group_rows_timestamp_global"].mean()
                ),
            }
            for target_name, target_column in TARGETS.items():
                base[f"ic_{target_name}"] = _corr(
                    local["score_raw"], local[target_column]
                )
            rows.append(base)
            for fraction in FRACTIONS:
                selected = stable_top(local, fraction)
                item = {
                    **{
                        key: value
                        for key, value in base.items()
                        if not key.startswith("ic_")
                    },
                    "fraction": float(fraction),
                    "rows": int(len(selected)),
                    "mean_gross_bps": float(
                        selected.execution_gross_ev_12h.mean() * 1e4
                    ),
                    "mean_cost_bps": float(
                        selected.execution_cost_return.mean() * 1e4
                    ),
                    "mean_net_bps": float(
                        selected.execution_net_ev_12h.mean() * 1e4
                    ),
                    "opportunity_0bps_precision": float(
                        selected.opportunity_gross_above_cost_0bps.mean()
                    ),
                    "opportunity_25bps_precision": float(
                        selected.opportunity_gross_above_cost_25bps.mean()
                    ),
                    "opportunity_0bps_lift_vs_month": float(
                        selected.opportunity_gross_above_cost_0bps.mean()
                        / overall_0
                    ),
                    "opportunity_25bps_lift_vs_month": float(
                        selected.opportunity_gross_above_cost_25bps.mean()
                        / overall_25
                    ),
                    "long_share": float(selected.side_name.eq("long").mean()),
                    "timestamp_hhi": _hhi(selected["__ts__"]),
                    "asset_hhi": _hhi(selected["__symbol__"]),
                    "mean_candidate_group_size": float(
                        selected["base_group_rows_timestamp_global"].mean()
                    ),
                }
                for target_name, target_column in TARGETS.items():
                    item[f"ic_{target_name}"] = _corr(
                        selected["score_raw"], selected[target_column]
                    )
                rows.append(item)
    return pd.DataFrame(rows)


def decile_monotonicity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    outcomes = (
        "execution_mfe_return_12h",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        "opportunity_gross_above_cost_0bps",
        "opportunity_gross_above_cost_25bps",
        "exit_is_trailing",
        "exit_is_timeout",
        "exit_is_full_stop",
        "exit_is_adverse_exit",
    )
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        scopes = [("pooled_global", month_rows)]
        scopes.extend(
            (f"side_{side}", rows)
            for side, rows in month_rows.groupby("side_name", sort=True)
        )
        for scope, local in scopes:
            ranked = local.copy()
            # Decile 9 is always the highest score.  Ranking first makes ties
            # deterministic in current row order, which is already identity
            # sorted by the source panel.
            rank = ranked["score_raw"].rank(method="first", pct=True)
            ranked["score_decile"] = (
                rank.mul(10.0).clip(0.0, 9.999999).astype(int)
            )
            grouped = ranked.groupby("score_decile", sort=True, observed=True)
            local_cells: list[dict[str, Any]] = []
            for decile, group in grouped:
                row: dict[str, Any] = {
                    "candidate_month": str(month),
                    "scope": scope,
                    "score_decile": int(decile),
                    "rows": int(len(group)),
                    "score_mean": float(group.score_raw.mean()),
                    "score_std": float(group.score_raw.std()),
                }
                for outcome in outcomes:
                    values = pd.to_numeric(group[outcome], errors="raise").astype(float)
                    row[f"{outcome}__mean"] = float(values.mean())
                    row[f"{outcome}__se"] = float(
                        values.std(ddof=1) / math.sqrt(max(len(values), 1))
                    )
                cells.append(row)
                local_cells.append(row)
            local_frame = pd.DataFrame(local_cells).sort_values("score_decile")
            summary: dict[str, Any] = {
                "candidate_month": str(month),
                "scope": scope,
                "deciles": int(len(local_frame)),
            }
            for outcome in outcomes:
                means = local_frame[f"{outcome}__mean"]
                summary[f"{outcome}__decile_spearman"] = _corr(
                    local_frame["score_decile"], means
                )
                summary[f"{outcome}__adjacent_violations"] = int(
                    means.diff().dropna().lt(0.0).sum()
                )
                summary[f"{outcome}__top_minus_bottom"] = float(
                    means.iloc[-1] - means.iloc[0]
                )
            summaries.append(summary)
    return pd.DataFrame(cells), pd.DataFrame(summaries)


def score_dispersion(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["candidate_month", "side_name", "__ts__"], sort=True)
        .agg(
            rows=("candidate_id", "size"),
            score_mean=("score_raw", "mean"),
            score_std=("score_raw", "std"),
            score_min=("score_raw", "min"),
            score_max=("score_raw", "max"),
            top40_cutoff=("base_top40_cutoff_timestamp_side", "first"),
        )
        .reset_index()
    )
    grouped["score_range"] = grouped["score_max"] - grouped["score_min"]
    summaries: list[dict[str, Any]] = []
    for (month, side), local in grouped.groupby(
        ["candidate_month", "side_name"], sort=True
    ):
        row: dict[str, Any] = {
            "candidate_month": str(month),
            "side_name": str(side),
            "timestamps": int(len(local)),
            "candidate_rows_mean": float(local.rows.mean()),
        }
        for field in ("score_mean", "score_std", "score_range", "top40_cutoff"):
            for quantile in (0.10, 0.50, 0.90):
                row[f"{field}_p{int(quantile * 100):02d}"] = float(
                    local[field].quantile(quantile)
                )
        summaries.append(row)
    return pd.DataFrame(summaries)


def candidate_group_slices(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, local in frame.groupby("candidate_month", sort=True):
        work = local.copy()
        work["candidate_group_quartile"] = pd.qcut(
            work["base_group_rows_timestamp_global"].rank(method="first"),
            q=4,
            labels=False,
        )
        for quartile, group in work.groupby(
            "candidate_group_quartile", sort=True, observed=True
        ):
            rows.append(
                {
                    "candidate_month": str(month),
                    "candidate_group_quartile": int(quartile),
                    "rows": int(len(group)),
                    "candidate_group_rows_mean": float(
                        group.base_group_rows_timestamp_global.mean()
                    ),
                    "native_ic": _corr(
                        group.score_raw, group["__first_touch_target_soft__"]
                    ),
                    "net_ic": _corr(
                        group.score_raw, group.execution_net_ev_12h
                    ),
                    "opportunity_rate": float(
                        group.opportunity_gross_above_cost_0bps.mean()
                    ),
                    "net_bps": float(group.execution_net_ev_12h.mean() * 1e4),
                }
            )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: sha256(path) for path in paths}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    panel_manifest = json.loads((PANEL_ROOT / "manifest.json").read_text())
    if (
        panel_manifest.get("schema")
        != "canonical_opportunity_payoff_trust_panel_v2"
        or panel_manifest.get("rows") != 509_868
    ):
        raise ValueError("audited canonical v2 panel required")
    if (
        sha256(PANEL_ROOT / "manifest.json")
        != (PANEL_ROOT / "manifest.sha256").read_text().split()[0]
    ):
        raise ValueError("canonical panel detached manifest checksum fails")
    required = {
        "candidate_id",
        "candidate_month",
        "side_name",
        "__symbol__",
        "__ts__",
        "base_oof_score",
        "base_group_rows_timestamp_global",
        "base_top40_cutoff_timestamp_side",
        *TARGETS.values(),
        "execution_cost_return",
        "execution_exit_class",
        "exit_is_trailing",
        "exit_is_timeout",
        "exit_is_full_stop",
        "exit_is_adverse_exit",
    }
    panel = pd.read_parquet(PANEL_ROOT / "panel.parquet", columns=sorted(required))
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True, errors="raise")
    panel["score_raw"] = pd.to_numeric(panel.pop("base_oof_score"), errors="raise")
    panel = panel.sort_values(
        ["candidate_month", "__ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if len(panel) != 509_868 or panel["candidate_id"].duplicated().any():
        raise ValueError("canonical diagnostic identity changed")

    tail = full_and_tail_metrics(panel)
    deciles, monotonicity = decile_monotonicity(panel)
    dispersion = score_dispersion(panel)
    group_slices = candidate_group_slices(panel)
    counterfactuals: list[pd.DataFrame] = []
    selected_summaries: list[pd.DataFrame] = []
    for fraction in FRACTIONS:
        summary, selected = selected_month_components(panel, fraction=fraction)
        summary["fraction"] = fraction
        selected_summaries.append(summary)
        counter = rank_conversion_counterfactual(panel, selected, bins=100)
        counter["fraction"] = fraction
        counterfactuals.append(counter)
    selected_summary = pd.concat(selected_summaries, ignore_index=True)
    counterfactual = pd.concat(counterfactuals, ignore_index=True)

    feb_mar = counterfactual.loc[
        counterfactual["from_month"].astype(str).eq("2025-02")
        & counterfactual["to_month"].astype(str).eq("2025-03")
    ].sort_values("fraction")
    top10 = tail.loc[
        tail["scope"].eq("pooled_global") & tail["fraction"].eq(0.10)
    ].sort_values("candidate_month")
    findings = {
        "pooled_global_top10_by_month": top10[
            [
                "candidate_month",
                "rows",
                "mean_net_bps",
                "opportunity_0bps_precision",
                "opportunity_0bps_lift_vs_month",
                "ic_native_alpha_soft_24h",
                "ic_exact_net_12h",
                "long_share",
                "timestamp_hhi",
                "asset_hhi",
            ]
        ].to_dict(orient="records"),
        "february_to_march_by_fraction": feb_mar[
            [
                "fraction",
                "ordering_composition_effect_bps",
                "rank_to_economics_conversion_effect_bps",
            ]
        ].to_dict(orient="records"),
        "interpretation_contract": [
            "Full-population rank IC, truncated-tail IC and realized tail EV answer different questions and must be reported together.",
            "A positive or improving IC cannot promote a score when pooled global tail opportunity prevalence or conditional payoff deteriorates.",
            "Timestamp, side and asset concentration are explanations/risks, never alternate per-group selection policies.",
        ],
    }

    temporary = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    outputs = {
        "full_and_tail_metrics.csv": tail,
        "score_decile_economics.csv": deciles,
        "decile_monotonicity.csv": monotonicity,
        "score_dispersion.csv": dispersion,
        "candidate_group_slices.csv": group_slices,
        "selected_month_components.csv": selected_summary,
        "rank_conversion_counterfactual.csv": counterfactual,
    }
    for name, frame in outputs.items():
        _write_csv(temporary / name, frame)
    (temporary / "findings.json").write_text(
        json.dumps(_safe(findings), indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema": "canonical_base_ic_ev_tail_diagnostic_v1",
        "status": "COMPLETED_DIAGNOSTIC_NO_MODEL_PROMOTION",
        "rows": int(len(panel)),
        "source": {
            "panel_manifest": str(PANEL_ROOT / "manifest.json"),
            "panel_manifest_sha256": sha256(PANEL_ROOT / "manifest.json"),
            "panel_sha256": sha256(PANEL_ROOT / "panel.parquet"),
            "identity_sha256": panel_manifest["identity_sha256"],
        },
        "contracts": {
            "selection": "one pooled global monthly top 1/5/10/20% with candidate_id tie-break; never timestamp/side quotas",
            "rank_bridge": "100 side x within-month score-rank cells with symmetric two-state ordering/conversion interaction allocation",
            "uncertainty": "decile standard errors are descriptive IID standard errors; temporal dependence means they are not promotion confidence intervals",
            "scope": "diagnostic attribution only; no model fit, threshold tuning, portfolio replay or promotion",
        },
        "outputs_sha256": _hashes(
            [temporary / name for name in [*outputs, "findings.json"]]
        ),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n"
    )
    os.replace(temporary, OUT)
    print(
        json.dumps(
            {
                "output": str(OUT),
                "rows": len(panel),
                "top10": findings["pooled_global_top10_by_month"],
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
