#!/usr/bin/env python3
"""Measure the *non-tradable* upper ceiling of exact execution-EV labels.

An oracle ranks candidates by their realized 12-hour net return.  This is not a
backtest, signal, model-selection criterion, or promotion evidence.  Its only
purpose is to answer a prior question: under the frozen exit/cost contract, is
there enough ex-post opportunity for a predictor to recover?

The runner is deliberately strict about label provenance.  It accepts either
individual canonical label panels with their manifests, or the frozen-common
universe transfer ledger with its summary.  It verifies the output hashes,
12-hour policy horizon, simulator, policy hash, current-spread lineage, and
universal side-parent fallback before reading any candidate returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.universe import _normalize_symbol


REQUIRED_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "candidate_id",
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_cost_return",
)
POLICY_SHA = "aed39b3474f06a2134ed814bccaf41e0a3fd54bd8194108dfa251f6abcdce301"
HORIZON_MINUTES = 720
SPREAD_BUCKETS = (-np.inf, 10.0, 25.0, 70.0, 150.0, np.inf)
SPREAD_BUCKET_LABELS = ("<=10", "10-25", "25-70", "70-150", ">150")
MARGINS_BPS = (0.0, 25.0, 50.0)
TOP_FRACTIONS = (0.10, 0.05, 0.02)


@dataclass(frozen=True)
class Panel:
    """A verified label panel and its diagnostic scope."""

    name: str
    labels: Path
    provenance: Mapping[str, Any]
    scope: str


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _manifest_policy_contract(manifest: Mapping[str, Any], *, source: Path) -> None:
    exit_contract = manifest.get("exit_policy_contract")
    accounting = manifest.get("accounting")
    geometry = manifest.get("geometry")
    _require(manifest.get("schema") == "execution_ev_deployed_policy_1m_labels_v1",
             f"unexpected label schema: {source}")
    _require(isinstance(exit_contract, dict), f"missing exit policy contract: {source}")
    _require(isinstance(accounting, dict), f"missing accounting contract: {source}")
    _require(isinstance(geometry, dict), f"missing geometry contract: {source}")
    _require(exit_contract.get("horizon_minutes") == HORIZON_MINUTES,
             f"non-12h label horizon: {source}")
    _require(exit_contract.get("source_policy_sha256") == POLICY_SHA,
             f"unexpected frozen policy hash: {source}")
    _require("simple_policy_optimiser.simulate_and_score" in str(exit_contract.get("simulator")),
             f"unexpected label simulator: {source}")
    _require("simple_policy_optimiser.simulate_and_score" in str(accounting.get("simulator")),
             f"unexpected accounting simulator: {source}")
    _require(geometry.get("fallback_rate") == 1.0,
             f"label geometry is not universally side-parent fallback: {source}")
    _require(
        str(accounting.get("cost_return"))
        == "fee return; spread drag is embedded in gross return",
        f"gross/cost accounting mismatch: {source}",
    )
    _require(bool(accounting.get("spread_baseline_sha256")),
             f"missing current spread baseline lineage: {source}")


def verify_label_panel(*, name: str, labels: Path, manifest_path: Path) -> Panel:
    """Verify a direct canonical label panel before exposing any outcomes."""
    manifest = _json(manifest_path)
    _manifest_policy_contract(manifest, source=manifest_path)
    output = manifest.get("output")
    _require(isinstance(output, dict), f"missing label output lineage: {manifest_path}")
    _require(output.get("sha256") == _sha(labels), f"label hash mismatch: {labels}")
    _require(int(output.get("rows", -1)) > 0, f"empty label panel: {labels}")
    return Panel(
        name=name,
        labels=labels,
        scope="canonical_current_spread_label_panel",
        provenance={
            "labels": str(labels),
            "labels_sha256": _sha(labels),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha(manifest_path),
            "policy_sha256": manifest["exit_policy_contract"]["source_policy_sha256"],
            "spread_baseline_sha256": manifest["accounting"]["spread_baseline_sha256"],
        },
    )


def verify_transfer_panel(*, name: str, labels: Path, summary_path: Path) -> Panel:
    """Verify a composed common-universe ledger through its source contracts."""
    summary = _json(summary_path)
    artifacts = summary.get("artifacts")
    target = summary.get("target")
    _require(isinstance(artifacts, dict), f"missing artifact hashes: {summary_path}")
    _require(isinstance(target, dict), f"missing target contract: {summary_path}")
    _require(
        artifacts.get(labels.name) == _sha(labels),
        f"transfer label hash mismatch: {labels}",
    )
    _require(target.get("horizon_minutes") == HORIZON_MINUTES,
             f"transfer label horizon is not 12h: {summary_path}")
    _require(target.get("policy_sha256") == POLICY_SHA,
             f"transfer policy hash mismatch: {summary_path}")
    _require("simple_policy_optimiser.simulate_and_score" in str(target.get("simulator")),
             f"transfer simulator mismatch: {summary_path}")
    geometry = target.get("geometry")
    _require(isinstance(geometry, dict), f"missing transfer target geometry: {summary_path}")
    _require(geometry.get("mode") == "external_current_spread_counterfactual",
             f"transfer labels are not current-spread counterfactuals: {summary_path}")
    panels = geometry.get("panels")
    _require(isinstance(panels, list) and panels, f"missing transfer source panels: {summary_path}")
    for panel in panels:
        checks = panel.get("checks") if isinstance(panel, dict) else None
        _require(isinstance(checks, dict) and all(bool(checks.get(key)) for key in (
            "horizon_minutes", "policy_sha256", "side_parent_fallback", "simulator", "spread_sha256"
        )), f"failed source label contract in transfer ledger: {summary_path}")
        manifest_raw = panel.get("manifest")
        manifest_sha = panel.get("manifest_sha256")
        _require(isinstance(manifest_raw, str) and isinstance(manifest_sha, str),
                 f"missing source-manifest lineage: {summary_path}")
        source_manifest = Path(manifest_raw)
        _require(source_manifest.exists() and _sha(source_manifest) == manifest_sha,
                 f"source manifest hash mismatch: {source_manifest}")
        _manifest_policy_contract(_json(source_manifest), source=source_manifest)
    return Panel(
        name=name,
        labels=labels,
        scope="frozen_common30_transfer_label_ledger",
        provenance={
            "labels": str(labels),
            "labels_sha256": _sha(labels),
            "summary": str(summary_path),
            "summary_sha256": _sha(summary_path),
            "policy_sha256": target["policy_sha256"],
            "source_panel_count": len(panels),
        },
    )


def load_panel(panel: Panel) -> pd.DataFrame:
    frame = pd.read_parquet(panel.labels, columns=list(REQUIRED_COLUMNS))
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    _require(not missing, f"missing canonical target columns in {panel.labels}: {missing}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    _require(frame["side_name"].isin(("long", "short")).all(),
             f"noncanonical side in {panel.labels}")
    _require(not frame.duplicated(["__ts__", "__symbol__", "side_name", "candidate_id"]).any(),
             f"duplicate candidate identity in {panel.labels}")
    values = frame[["execution_gross_ev_12h", "execution_net_ev_12h", "execution_cost_return"]]
    _require(np.isfinite(values.to_numpy(np.float64)).all(),
             f"nonfinite economic label in {panel.labels}")
    # This is exact in the simulator's accounting.  A tight tolerance catches a
    # silently changed target or accidental double-cost subtraction.
    _require(
        np.allclose(
            frame["execution_gross_ev_12h"] - frame["execution_cost_return"],
            frame["execution_net_ev_12h"],
            rtol=0.0,
            atol=1e-10,
        ),
        f"gross-cost-net reconciliation failed: {panel.labels}",
    )
    frame["panel"] = panel.name
    frame["period_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    # Period labels have no timezone representation.  The timestamp is already
    # canonical UTC, so remove only the UTC annotation for this display/group
    # label rather than letting pandas silently choose host-local time.
    frame["period_week"] = (
        frame["__ts__"].dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    )
    return frame


def load_spread_baseline(path: Path, *, threshold_bps: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    needed = {"symbol", "average_spread_bps"}
    _require(needed.issubset(frame.columns), f"invalid spread baseline: {path}")
    result = pd.DataFrame({
        "symbol_norm": frame["symbol"].astype(str).map(_normalize_symbol),
        "baseline_average_spread_bps": pd.to_numeric(frame["average_spread_bps"], errors="coerce"),
    })
    _require(result["symbol_norm"].ne("").all(), f"blank normalized symbol: {path}")
    _require(not result.duplicated("symbol_norm").any(), f"duplicate normalized spread symbol: {path}")
    _require(np.isfinite(result["baseline_average_spread_bps"]).all(), f"nonfinite spread: {path}")
    result["inference_eligible"] = result["baseline_average_spread_bps"].le(threshold_bps)
    return result


def annotate_spread(frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["symbol_norm"] = result["__symbol__"].map(_normalize_symbol)
    result = result.merge(baseline, on="symbol_norm", how="left", validate="many_to_one")
    result["inference_eligible"] = result["inference_eligible"].fillna(False).astype(bool)
    result["spread_bucket"] = pd.cut(
        result["baseline_average_spread_bps"],
        bins=list(SPREAD_BUCKETS),
        labels=list(SPREAD_BUCKET_LABELS),
        include_lowest=True,
        right=True,
    ).astype("string").fillna("missing_baseline")
    return result


def _economics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "candidate_rows": 0,
            "mean_gross_bps": None,
            "mean_cost_bps": None,
            "mean_net_bps": None,
            "sum_net_return": None,
            "positive_net_rate": None,
        }
    return {
        "candidate_rows": int(len(rows)),
        "mean_gross_bps": float(rows["execution_gross_ev_12h"].mean() * 1e4),
        "mean_cost_bps": float(rows["execution_cost_return"].mean() * 1e4),
        "mean_net_bps": float(rows["execution_net_ev_12h"].mean() * 1e4),
        "sum_net_return": float(rows["execution_net_ev_12h"].sum()),
        "positive_net_rate": float(rows["execution_net_ev_12h"].gt(0.0).mean()),
    }


def opportunity_rows(frame: pd.DataFrame, *, panel: str, universe: str) -> list[dict[str, Any]]:
    """Return economically feasible opportunity rates, never a model metric."""
    rows: list[dict[str, Any]] = []
    groupings: Iterable[tuple[str, pd.Series | None]] = (
        ("overall", None),
        ("month", frame["period_month"]),
        ("week", frame["period_week"]),
        ("side", frame["side_name"]),
        ("spread_bucket", frame["spread_bucket"]),
    )
    for grouping, values in groupings:
        iterator = [("all", frame)] if values is None else frame.groupby(values, observed=True, sort=True)
        for value, group in iterator:
            for margin_bps in MARGINS_BPS:
                margin = margin_bps / 1e4
                above_net = group["execution_net_ev_12h"].gt(margin)
                above_gross = group["execution_gross_ev_12h"].gt(
                    group["execution_cost_return"] + margin
                )
                # They must agree by the required exact gross-cost-net check.
                _require(above_net.equals(above_gross), "net and gross threshold mismatch")
                rows.append({
                    "panel": panel,
                    "universe": universe,
                    "grouping": grouping,
                    "group": str(value),
                    "margin_bps": margin_bps,
                    **_economics(group),
                    "gross_above_cost_plus_margin_rate": float(above_gross.mean()),
                    "net_above_margin_rate": float(above_net.mean()),
                    "gross_above_cost_plus_margin_rows": int(above_gross.sum()),
                })
    return rows


def _topk_metrics(
    candidates: pd.DataFrame,
    *,
    panel: str,
    universe: str,
    selection_scope: str,
    group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        count = max(1, int(np.ceil(len(candidates) * fraction))) if len(candidates) else 0
        selected = candidates.nlargest(count, "execution_net_ev_12h")
        rows.append({
            "panel": panel,
            "universe": universe,
            "selection_scope": selection_scope,
            "group": group,
            "top_fraction": fraction,
            "selected_rows": int(len(selected)),
            "selected_fraction": len(selected) / max(len(candidates), 1),
            **_economics(selected),
        })
    return rows


def oracle_topk_rows(frame: pd.DataFrame, *, panel: str, universe: str) -> list[dict[str, Any]]:
    """Global ceiling plus clearly separated local diagnostic ceilings."""
    rows = _topk_metrics(frame, panel=panel, universe=universe,
                         selection_scope="one_global_book", group="all")
    for grouping, values in (
        ("month_local_oracle", frame["period_month"]),
        ("week_local_oracle", frame["period_week"]),
        ("side_local_oracle", frame["side_name"]),
        ("spread_bucket_local_oracle", frame["spread_bucket"]),
    ):
        for value, group in frame.groupby(values, observed=True, sort=True):
            rows.extend(_topk_metrics(group, panel=panel, universe=universe,
                                      selection_scope=grouping, group=str(value)))
    return rows


def variable_admission_rows(frame: pd.DataFrame, *, panel: str, universe: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for margin_bps in MARGINS_BPS:
        admitted = frame.loc[frame["execution_net_ev_12h"].gt(margin_bps / 1e4)]
        rows.append({
            "panel": panel,
            "universe": universe,
            "margin_bps": margin_bps,
            "admitted_rows": int(len(admitted)),
            "admitted_fraction": len(admitted) / max(len(frame), 1),
            **_economics(admitted),
        })
    return rows


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite oracle ceiling artifact")
    _require(len(args.labels) == len(args.manifests) == len(args.panel_names),
             "--labels, --manifests and --panel-names must have identical counts")
    panel_list = [
        verify_label_panel(name=name, labels=labels, manifest_path=manifest)
        for name, labels, manifest in zip(args.panel_names, args.labels, args.manifests)
    ]
    if args.transfer_labels or args.transfer_summary or args.transfer_name:
        _require(bool(args.transfer_labels and args.transfer_summary and args.transfer_name),
                 "transfer labels, summary and name must be supplied together")
        panel_list.append(verify_transfer_panel(
            name=args.transfer_name,
            labels=args.transfer_labels,
            summary_path=args.transfer_summary,
        ))
    _require(panel_list, "at least one verified panel is required")
    _require(len({panel.name for panel in panel_list}) == len(panel_list), "duplicate panel name")

    baseline = load_spread_baseline(args.spread_baseline, threshold_bps=args.spread_threshold_bps)
    panel_meta: list[dict[str, Any]] = []
    opportunity: list[dict[str, Any]] = []
    topk: list[dict[str, Any]] = []
    variable: list[dict[str, Any]] = []
    for panel in panel_list:
        annotated = annotate_spread(load_panel(panel), baseline)
        missing_baseline = int(annotated["baseline_average_spread_bps"].isna().sum())
        panel_meta.append({
            "name": panel.name,
            "scope": panel.scope,
            "rows": int(len(annotated)),
            "min_timestamp": str(annotated["__ts__"].min()),
            "max_timestamp": str(annotated["__ts__"].max()),
            "missing_spread_baseline_rows": missing_baseline,
            "inference_eligible_rows": int(annotated["inference_eligible"].sum()),
            "provenance": dict(panel.provenance),
        })
        for universe, subset in (
            ("all_candidates", annotated),
            ("inference_eligible_average_spread_le_threshold", annotated.loc[annotated["inference_eligible"]]),
        ):
            _require(not subset.empty, f"empty universe {universe} for {panel.name}")
            opportunity.extend(opportunity_rows(subset, panel=panel.name, universe=universe))
            topk.extend(oracle_topk_rows(subset, panel=panel.name, universe=universe))
            variable.extend(variable_admission_rows(subset, panel=panel.name, universe=universe))

    args.output_dir.mkdir(parents=True)
    paths = {
        "opportunity_rates": _write_csv(pd.DataFrame(opportunity), args.output_dir / "opportunity_rates.csv"),
        "oracle_topk": _write_csv(pd.DataFrame(topk), args.output_dir / "oracle_topk_metrics.csv"),
        "variable_admission": _write_csv(pd.DataFrame(variable), args.output_dir / "variable_admission_metrics.csv"),
    }
    manifest = {
        "schema": "execution_ev_oracle_opportunity_ceiling_v1",
        "status": "non_tradable_ex_post_oracle_diagnostic_not_promotion_evidence",
        "purpose": (
            "Upper bound on available realized 12h opportunity under the frozen "
            "exit/cost contract. It ranks on future realized labels and cannot "
            "be used as a score, policy, HPO objective, or backtest."
        ),
        "contract": {
            "rank_target": "realized execution_net_ev_12h (future outcome; oracle only)",
            "topk": "one pooled global book across side and timestamp; local slices are explicitly labelled local oracle diagnostics",
            "variable_admission": "realized net > 0/25/50 bps; oracle-only opportunity rate",
            "gross_cost_net": "gross return includes deployed spread drag; cost_return is fee; net = gross - cost",
            "spread_universe": (
                "current frozen average-spread blacklist, normalized exactly as "
                "inference universe.py; missing baseline is fail-closed ineligible"
            ),
            "spread_threshold_bps": args.spread_threshold_bps,
            "policy_sha256": POLICY_SHA,
            "horizon_minutes": HORIZON_MINUTES,
        },
        "inputs": {
            "spread_baseline": str(args.spread_baseline),
            "spread_baseline_sha256": _sha(args.spread_baseline),
            "panels": panel_meta,
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--labels", type=Path, action="append", default=[])
    result.add_argument("--manifests", type=Path, action="append", default=[])
    result.add_argument("--panel-names", action="append", default=[])
    result.add_argument("--transfer-labels", type=Path)
    result.add_argument("--transfer-summary", type=Path)
    result.add_argument("--transfer-name")
    result.add_argument(
        "--spread-baseline",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/spread_model/per_asset_spread_baseline_latest.csv"),
    )
    result.add_argument("--spread-threshold-bps", type=float, default=70.0)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps({name: str(path) for name, path in run(args).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
