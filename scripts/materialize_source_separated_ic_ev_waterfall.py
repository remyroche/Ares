#!/usr/bin/env python3
"""Materialize the IC-to-execution-EV waterfall without mixing evidence sources.

Every ledger in the immutable historical conversion manifest is evaluated
independently. This is deliberately a diagnostic: it consumes only the
upstream-declared ``score_*`` streams, applies no new mapping, changes no
model/policy, and makes no promotion decision. Some declared streams are raw
alpha/direct outputs and others are upstream expected-EV or residual
components; their names and source contracts remain explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1"
)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
WATERFALL_IDENTITY_ATOL = 1e-7
TARGETS = {
    "legacy_native24_base_target": "__first_touch_target_soft__",
    "exact_12h_mfe_ceiling": "execution_mfe_return_12h",
    "exact_12h_gross": "execution_gross_ev_12h",
    "exact_cost": "execution_cost_return",
    "exact_12h_net": "execution_net_ev_12h",
}
IDENTITY_COLUMNS = ("candidate_id", "side_name", "__symbol__", "__ts__")
TIE_IDENTITY = "candidate_id"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def score_columns(frame: pd.DataFrame) -> list[str]:
    """Return declared score streams; explicitly mapped fields fail closed."""

    columns = sorted(column for column in frame if column.startswith("score_"))
    illegal = [column for column in columns if "map" in column.lower()]
    if illegal:
        raise ValueError(f"mapped score column is forbidden: {illegal}")
    if not columns:
        raise ValueError("source ledger has no raw score_* columns")
    return columns


def score_role(column: str) -> str:
    if column == "score_base_alpha":
        return "raw_base_alpha"
    if column == "score_direct_execution_ev":
        return "direct_execution_ev_oof"
    if column == "score_residual_delta_ev":
        return "residual_delta_component"
    if column in {"score_base_expected_ev", "score_residual_expected_ev"}:
        return "upstream_expected_ev_stream"
    return "other_declared_score_stream"


def _finite_pair(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    local = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    return local.loc[np.isfinite(local.left) & np.isfinite(local.right)]


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    local = _finite_pair(left, right)
    if len(local) < 3 or local.left.nunique() < 2 or local.right.nunique() < 2:
        return np.nan
    value = local.left.corr(local.right, method="spearman")
    return float(value) if np.isfinite(value) else np.nan


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    """Raw-score descending, candidate-ID ascending tie resolution."""

    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    values = pd.to_numeric(frame[score], errors="raise").to_numpy(float)
    order = np.lexsort((frame[TIE_IDENTITY].astype(str).to_numpy(), -values))
    return frame.iloc[order[:count]].copy()


def scopes(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "pooled_global", frame
    for side, local in frame.groupby("side_name", sort=True, observed=True):
        yield f"side_{side}", local


def full_ic(frame: pd.DataFrame, *, source_family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in scopes(month_rows):
            for target, column in TARGETS.items():
                if column not in local:
                    continue
                rows.append({
                    "source_family": source_family,
                    "score": score,
                    "candidate_month": str(month),
                    "scope": scope,
                    "rows": int(len(local)),
                    "target": target,
                    "rank_ic": rank_ic(local[score], local[column]),
                })
    return pd.DataFrame(rows)


def _cvar05(values: pd.Series) -> float:
    local = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if local.empty:
        return np.nan
    return float(local.iloc[: max(1, int(math.ceil(len(local) * 0.05)))].mean())


def tail_metrics(frame: pd.DataFrame, *, source_family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in scopes(month_rows):
            for fraction in TOP_FRACTIONS:
                selected = stable_top(local, score, fraction)
                gross = pd.to_numeric(selected.execution_gross_ev_12h, errors="raise")
                mfe = pd.to_numeric(selected.execution_mfe_return_12h, errors="raise")
                cost = pd.to_numeric(selected.execution_cost_return, errors="raise")
                net = pd.to_numeric(selected.execution_net_ev_12h, errors="raise")
                opportunity = selected.opportunity_gross_above_cost_0bps.astype(bool)
                positive = net.gt(0.0)
                loss = net.lt(0.0)
                rows.append({
                    "source_family": source_family, "score": score,
                    "candidate_month": str(month), "scope": scope,
                    "fraction": float(fraction), "candidate_rows": int(len(local)),
                    "selected_rows": int(len(selected)),
                    # The ledger materializer proves this indicator is exactly
                    # net > 0, so it must never be interpreted as an
                    # independent opportunity event.
                    "net_positive_rate_alias_named_opportunity": float(opportunity.mean()),
                    "precision_positive_net": float(positive.mean()),
                    "loss_rate": float(loss.mean()), "cvar05_net_bps": _cvar05(net) * 1e4,
                    "tail_net_rank_ic": rank_ic(selected[score], net),
                    "mean_mfe_bps": float(mfe.mean() * 1e4),
                    "mean_gross_bps": float(gross.mean() * 1e4),
                    "mean_cost_bps": float(cost.mean() * 1e4),
                    "mean_net_bps": float(net.mean() * 1e4),
                    "mfe_ceiling_to_gross_gap_bps": float((mfe.mean() - gross.mean()) * 1e4),
                    "gross_to_net_explicit_cost_gap_bps": float((gross.mean() - net.mean()) * 1e4),
                })
    return pd.DataFrame(rows)


def score_compression(frame: pd.DataFrame, *, source_family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in scopes(month_rows):
            values = pd.to_numeric(local[score], errors="raise")
            top = stable_top(local, score, 0.10)[score]
            rows.append({
                "source_family": source_family, "score": score,
                "candidate_month": str(month), "scope": scope, "rows": int(len(local)),
                "unique_score_levels": int(values.nunique()),
                "unique_score_fraction": float(values.nunique() / len(values)),
                "score_std": float(values.std(ddof=0)),
                "score_iqr": float(values.quantile(.75) - values.quantile(.25)),
                "score_span": float(values.max() - values.min()),
                "top10_unique_score_levels": int(pd.Series(top).nunique()),
                "top10_score_span": float(pd.Series(top).max() - pd.Series(top).min()),
            })
    return pd.DataFrame(rows)


def response_20bin(frame: pd.DataFrame, *, source_family: str, score: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in scopes(month_rows):
            work = local.copy()
            order = np.lexsort((work[TIE_IDENTITY].astype(str).to_numpy(), pd.to_numeric(work[score], errors="raise").to_numpy(float)))
            rank = np.empty(len(work), dtype=int); rank[order] = np.arange(len(work))
            work["score_rank_bin"] = np.minimum((rank * 20) // len(work), 19)
            means = work.groupby("score_rank_bin", sort=True, observed=True).agg(
                rows=(TIE_IDENTITY, "size"), mean_mfe=("execution_mfe_return_12h", "mean"),
                mean_gross=("execution_gross_ev_12h", "mean"), mean_cost=("execution_cost_return", "mean"),
                mean_net=("execution_net_ev_12h", "mean"),
                net_positive_rate_alias_named_opportunity=("opportunity_gross_above_cost_0bps", "mean"),
            ).reset_index()
            for row in means.itertuples(index=False):
                cells.append({"source_family": source_family, "score": score, "candidate_month": str(month), "scope": scope,
                              "score_rank_bin": int(row.score_rank_bin), "rows": int(row.rows),
                              "mean_mfe_bps": float(row.mean_mfe * 1e4), "mean_gross_bps": float(row.mean_gross * 1e4),
                              "mean_cost_bps": float(row.mean_cost * 1e4), "mean_net_bps": float(row.mean_net * 1e4),
                              "net_positive_rate_alias_named_opportunity": float(row.net_positive_rate_alias_named_opportunity)})
            net_means = means.mean_net.to_numpy(float)
            summaries.append({"source_family": source_family, "score": score, "candidate_month": str(month), "scope": scope,
                              "populated_bins": int(len(means)), "bin_to_net_rank_ic": rank_ic(means.score_rank_bin, means.mean_net),
                              "net_monotonicity_violations": int(np.sum(np.diff(net_means) < 0.0)),
                              "top_minus_bottom_net_bps": float((net_means[-1] - net_means[0]) * 1e4)})
    return pd.DataFrame(cells), pd.DataFrame(summaries)


def cutoff_ties(frame: pd.DataFrame, *, source_family: str, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True, observed=True):
        for scope, local in scopes(month_rows):
            for fraction in TOP_FRACTIONS:
                selected = stable_top(local, score, fraction)
                cutoff = float(selected[score].iloc[-1])
                above = local.loc[local[score].gt(cutoff)]
                tie = local.loc[local[score].eq(cutoff)]
                slots = len(selected) - len(above)
                chosen = pd.concat([above, tie.sort_values(TIE_IDENTITY, kind="mergesort").head(slots)])
                best = pd.concat([above, tie.nlargest(slots, "execution_net_ev_12h")])
                worst = pd.concat([above, tie.nsmallest(slots, "execution_net_ev_12h")])
                rows.append({"source_family": source_family, "score": score, "candidate_month": str(month), "scope": scope,
                             "fraction": float(fraction), "candidate_rows": int(len(local)), "selected_rows": int(len(selected)),
                             "cutoff_score": cutoff, "above_cutoff_rows": int(len(above)), "cutoff_tie_rows": int(len(tie)),
                             "slots_from_tie": int(slots), "candidate_id_tie_break_used": bool(len(tie) > slots),
                             "deterministic_net_bps": float(chosen.execution_net_ev_12h.mean() * 1e4),
                             "best_tie_net_bps": float(best.execution_net_ev_12h.mean() * 1e4),
                             "worst_tie_net_bps": float(worst.execution_net_ev_12h.mean() * 1e4),
                             "tie_sensitivity_bps": float((best.execution_net_ev_12h.mean() - worst.execution_net_ev_12h.mean()) * 1e4)})
    return pd.DataFrame(rows)


def fixed_composition(frame: pd.DataFrame, *, source_family: str, score: str) -> pd.DataFrame:
    """Decompose adjacent-month books across the full economic waterfall."""

    work = frame.copy()
    top_assets = set(work["__symbol__"].astype(str).value_counts().head(20).index)
    work["asset_bucket"] = np.where(work["__symbol__"].astype(str).isin(top_assets), work["__symbol__"].astype(str), "__other__")
    work["score_rank_decile"] = -1
    for _, local in work.groupby(
        ["candidate_month", "side_name"], sort=False, observed=True
    ):
        ordered = local.sort_values(
            [score, TIE_IDENTITY],
            ascending=[True, True],
            kind="stable",
        )
        decile = np.minimum(
            (np.arange(len(ordered), dtype=int) * 10) // len(ordered),
            9,
        )
        work.loc[ordered.index, "score_rank_decile"] = decile
    if work["score_rank_decile"].lt(0).any():
        raise AssertionError("deterministic score-rank cells were not assigned")
    work["cell"] = work.side_name.astype(str) + "|r" + work.score_rank_decile.astype(str) + "|" + work.asset_bucket.astype(str)
    components = {
        "net_positive_alias_pp": ("opportunity_gross_above_cost_0bps", 100.0),
        "mfe_ceiling_bps": ("execution_mfe_return_12h", 1e4),
        "deployed_gross_bps": ("execution_gross_ev_12h", 1e4),
        "explicit_cost_bps": ("execution_cost_return", 1e4),
        "exact_net_bps": ("execution_net_ev_12h", 1e4),
    }
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected: dict[str, pd.DataFrame] = {}
        for month, local in work.groupby("candidate_month", sort=True, observed=True):
            # This primary book remains global: side is a fixed-composition
            # cell, not a quota.  Side-local ranking only defines the cells.
            selected[str(month)] = stable_top(local, score, fraction)
        months = sorted(selected)
        for first, second in zip(months, months[1:]):
            a, b = selected[first], selected[second]
            aggregation_a: dict[str, tuple[str, str]] = {
                "rows_a": (TIE_IDENTITY, "size")
            }
            aggregation_b: dict[str, tuple[str, str]] = {
                "rows_b": (TIE_IDENTITY, "size")
            }
            for name, (column, _) in components.items():
                aggregation_a[f"{name}_a"] = (column, "mean")
                aggregation_b[f"{name}_b"] = (column, "mean")
            left = a.groupby("cell", observed=True).agg(**aggregation_a)
            right = b.groupby("cell", observed=True).agg(**aggregation_b)
            common = left.join(right, how="inner")
            if common.empty:
                continue
            wa, wb = common.rows_a / common.rows_a.sum(), common.rows_b / common.rows_b.sum()
            row: dict[str, Any] = {
                "source_family": source_family,
                "score": score,
                "fraction": float(fraction),
                "from_month": first,
                "to_month": second,
                "common_cells": int(len(common)),
                "from_common_mass": float(common.rows_a.sum() / len(a)),
                "to_common_mass": float(common.rows_b.sum() / len(b)),
            }
            for name, (_, scale) in components.items():
                a_value = float(np.dot(wa, common[f"{name}_a"]))
                b_value = float(np.dot(wb, common[f"{name}_b"]))
                a_under_b = float(np.dot(wb, common[f"{name}_a"]))
                row[f"from_fixed_composition_{name}"] = a_value * scale
                row[f"to_fixed_composition_{name}"] = b_value * scale
                row[f"composition_effect_{name}"] = (a_under_b - a_value) * scale
                row[f"within_cell_effect_{name}"] = (b_value - a_under_b) * scale
                row[f"fixed_composition_delta_{name}"] = (b_value - a_value) * scale
            row["from_fixed_composition_net_bps"] = row[
                "from_fixed_composition_exact_net_bps"
            ]
            row["to_fixed_composition_net_bps"] = row[
                "to_fixed_composition_exact_net_bps"
            ]
            row["composition_effect_bps"] = row[
                "composition_effect_exact_net_bps"
            ]
            row["within_cell_payoff_effect_bps"] = row[
                "within_cell_effect_exact_net_bps"
            ]
            row["fixed_composition_delta_bps"] = row[
                "fixed_composition_delta_exact_net_bps"
            ]
            rows.append(row)
    return pd.DataFrame(rows)


def validate_source(frame: pd.DataFrame, record: Mapping[str, Any]) -> None:
    required = {*IDENTITY_COLUMNS, "candidate_month", "execution_mfe_return_12h", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "opportunity_gross_above_cost_0bps"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{record['source_family']} missing required fields: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{record['source_family']} has duplicate four-field identities")
    if frame[TIE_IDENTITY].astype(str).duplicated().any():
        raise ValueError(f"{record['source_family']} has duplicate candidate IDs")
    if frame.source_family.nunique() != 1 or str(frame.source_family.iloc[0]) != str(record["source_family"]):
        raise ValueError("source family contract mismatch")
    for column in ("execution_mfe_return_12h", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy(float)).all():
            raise ValueError(f"non-finite economics: {column}")
    gross = pd.to_numeric(frame.execution_gross_ev_12h, errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame.execution_cost_return, errors="raise").to_numpy(float)
    net = pd.to_numeric(frame.execution_net_ev_12h, errors="raise").to_numpy(float)
    if not np.allclose(
        gross - cost,
        net,
        rtol=0.0,
        atol=WATERFALL_IDENTITY_ATOL,
    ):
        raise ValueError("waterfall identity failed: gross - explicit cost must equal net")
    named_opportunity = frame.opportunity_gross_above_cost_0bps.astype(bool).to_numpy()
    if not np.array_equal(named_opportunity, net > 0.0):
        raise ValueError("named opportunity label is not exactly the net-positive alias")
    for column in score_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite raw score: {column}")


def run(input_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "historical_score_economics_conversion_ledgers_v1":
        raise ValueError("requires immutable v1 historical conversion-ledger manifest")
    outputs: dict[str, list[pd.DataFrame]] = {key: [] for key in ("full_ic", "tails", "compression", "response_cells", "response_summary", "cutoff_ties", "fixed_composition")}
    input_records: list[dict[str, Any]] = []
    for record in manifest["ledgers"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"ledger hash mismatch: {path}")
        frame = pd.read_parquet(path)
        validate_source(frame, record)
        family = str(record["source_family"])
        declared_scores = score_columns(frame)
        input_records.append({"source_family": family, "path": str(path), "sha256": record["sha256"], "rows": int(len(frame)), "score_columns": declared_scores, "score_roles": {column: score_role(column) for column in declared_scores}, "evidence_tier": record["evidence_tier"], "promotion_eligible": bool(record["promotion_eligible"]), "exact_policy_parity": bool(record["exact_policy_parity"]), "path_frequency": record["path_frequency"], "cost_contract": record["cost_contract"], "historical_observed_spread": bool(record["historical_observed_spread"])})
        for score in score_columns(frame):
            outputs["full_ic"].append(full_ic(frame, source_family=family, score=score))
            outputs["tails"].append(tail_metrics(frame, source_family=family, score=score))
            outputs["compression"].append(score_compression(frame, source_family=family, score=score))
            cells, summary = response_20bin(frame, source_family=family, score=score)
            outputs["response_cells"].append(cells); outputs["response_summary"].append(summary)
            outputs["cutoff_ties"].append(cutoff_ties(frame, source_family=family, score=score))
            outputs["fixed_composition"].append(fixed_composition(frame, source_family=family, score=score))
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    for key, parts in outputs.items():
        result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        path = output_dir / f"{key}.parquet"; result.to_parquet(path, index=False, compression="zstd"); paths[key] = path
    report = {"schema": "source_separated_ic_ev_waterfall_audit_v1", "status": "DIAGNOSTIC_ONLY_NO_MAPPING_NO_PROMOTION",
              "contracts": {"source_separated": True, "identity": list(IDENTITY_COLUMNS), "declared_score_columns_only": True, "no_additional_mapping": True, "score_semantics": "score_base_alpha and score_direct_execution_ev are raw/OOF streams; residual delta is a component; base/residual expected-EV streams retain their upstream mapping/composition semantics and may contain plateaus", "selection": "per-month pooled global or side-local score book; score descending/candidate-ID ascending ties", "top_fractions": list(TOP_FRACTIONS), "response_bins": 20, "fixed_composition": "adjacent-month global top 1/5/10/20; side x deterministic side-local raw-score-rank-decile x source-static top20/other asset; opportunity/MFE/gross/cost/net decomposed separately", "mfe_semantics": "MFE is an upper-bound ceiling, not attainable gross", "legacy_base_target_semantics": "when present, __first_touch_target_soft__ is legacy native-24h alpha and is not a same-horizon execution target", "opportunity_semantics": "opportunity_gross_above_cost_0bps is exactly net>0 and is reported only as a named alias", "cost_semantics": "gross - explicit_cost = net within 1e-7 return-unit absolute tolerance; this covers observed float32 rounding <=5.9604645e-8 only; cost-contract/evidence-tier differences remain source-local, and canonical gross already embeds spread", "cost_rank_ic_semantics": "reported for completeness but may be rank-identical to gross/net and is not independent attribution evidence", "tie_bounds": "best/worst cutoff-tie economics use realized labels and are diagnostic sensitivity bounds only; deterministic selection remains candidate-ID based"},
              "input_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)}, "inputs": input_records,
              "outputs": {key: {"path": str(path), "sha256": sha256(path), "rows": int(len(pd.read_parquet(path)))} for key, path in paths.items()},
              "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
              "promotion_eligible": False}
    manifest_output = output_dir / "manifest.json"
    write_json(manifest_output, report)
    (output_dir / "manifest.sha256").write_text(
        sha256(manifest_output) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(safe(run(args.input_root, args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
