#!/usr/bin/env python3
"""Audit the exact v2 reliability cohort's IC-to-execution-EV conversion.

This runner is deliberately read-only.  It binds every reported score to the
same exact 12-hour deployed-exit labels and writes the selected identities, so
the native-target -> MFE -> gross -> cost -> net bridge cannot silently
reselect rows between stages.  It is a diagnostic only: no model, map,
threshold, action layer, or portfolio policy is fitted or changed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_ic_ev_diagnostic_20260730_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
FIXED_COST_RETURN = 0.01
N_BANDS = 10

# The score names stay source-specific.  In particular, the causal map is of
# ``raw_score``; it is not evidence that ``direct_q25_return`` was calibrated.
LAYERS: Mapping[str, Mapping[str, Any]] = {
    "base_alpha": {
        "score": "score_base_alpha",
        "score_semantics": "native 24h alpha; not a 12h expected-return calibration",
        "calibration_supported": False,
    },
    "residual_expected_ev": {
        "score": "score_residual_expected_ev",
        "score_semantics": "upstream residual expected-EV stream; diagnostic score scale only",
        "calibration_supported": False,
    },
    "direct_q25_return": {
        "score": "direct_q25_return",
        "score_semantics": "direct lower-quantile return; not a mean-EV calibration",
        "calibration_supported": False,
    },
    "raw_execution_score": {
        "score": "raw_score",
        "score_semantics": "raw v5 execution score; no newly fitted calibration",
        "calibration_supported": False,
    },
    "causal_mapped_raw_execution_score": {
        "score": "causal_pooled_21d",
        "eligible": "causal_pooled_21d_eligible",
        "score_semantics": "existing daily 21d pooled causal map of raw_score",
        "calibration_supported": True,
    },
}
IC_TARGETS: Mapping[str, str] = {
    "native_alpha_target_24h": "__first_touch_target_soft__",
    "exact_12h_mfe_ceiling": "execution_mfe_return_12h",
    "deployed_exit_gross_12h": "execution_gross_ev_12h",
    "explicit_row_cost_12h": "execution_cost_return",
    "exact_12h_net": "execution_net_ev_12h",
}
REQUIRED = {
    *IDENTITY,
    "candidate_month",
    *IC_TARGETS.values(),
    "target_economic_opportunity_hard",
    "target_net_positive",
    "target_positive_net_magnitude",
    "target_adverse_loss_magnitude",
    "exit_is_full_stop",
    "exit_is_timeout",
    "execution_exit_class",
    "__regime_source_execution_risk_score__",
    *(spec["score"] for spec in LAYERS.values()),
    "causal_pooled_21d_eligible",
}


class DiagnosticError(RuntimeError):
    pass


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


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    pair = pair.loc[np.isfinite(pair.left) & np.isfinite(pair.right)]
    if len(pair) < 3 or pair.left.nunique() < 2 or pair.right.nunique() < 2:
        return np.nan
    result = pair.left.corr(pair.right, method="spearman")
    return float(result) if np.isfinite(result) else np.nan


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    """One pooled cross-side/cross-timestamp book with a total tie order."""

    count = max(1, int(math.ceil(len(frame) * fraction)))
    values = pd.to_numeric(frame[score], errors="raise").to_numpy(float)
    order = np.lexsort((
        frame.side_name.astype(str).to_numpy(),
        frame.__symbol__.astype(str).to_numpy(),
        pd.to_datetime(frame.__ts__, utc=True, errors="raise").astype("int64").to_numpy(),
        frame.candidate_id.astype(str).to_numpy(),
        -values,
    ))
    return frame.iloc[order[:count]].copy()


def selected_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, IDENTITY].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True, errors="raise").astype(str)
    canonical = values.astype(str).sort_values(list(IDENTITY), kind="mergesort")
    return hashlib.sha256(canonical.to_csv(index=False).encode("utf-8")).hexdigest()


def layer_rows(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    spec = LAYERS[name]
    score = spec["score"]
    mask = pd.to_numeric(frame[score], errors="coerce").notna()
    eligible = spec.get("eligible")
    if eligible:
        mask &= frame[eligible].eq(True)
    result = frame.loc[mask].copy()
    if result.empty:
        raise DiagnosticError(f"{name} has no eligible score rows")
    return result


def periods(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for month, local in frame.groupby("candidate_month", sort=True, observed=True):
        yield str(month), local.copy()
    months = sorted(frame.candidate_month.astype(str).unique())
    if len(months) > 1:
        yield f"{months[0]}_to_{months[-1]}_all_available", frame.copy()


def bridge(frame: pd.DataFrame, layer: str, period: str) -> dict[str, Any]:
    score = LAYERS[layer]["score"]
    row: dict[str, Any] = {
        "layer": layer,
        "score": score,
        "period": period,
        "rows": int(len(frame)),
        "score_mean": float(frame[score].mean()),
        "score_std": float(frame[score].std(ddof=0)),
        "score_min": float(frame[score].min()),
        "score_max": float(frame[score].max()),
        "score_semantics": LAYERS[layer]["score_semantics"],
        "calibration_supported": bool(LAYERS[layer]["calibration_supported"]),
    }
    for target_name, target in IC_TARGETS.items():
        row[f"rank_ic__{target_name}"] = rank_ic(frame[score], frame[target])
    row["mean_mfe_bps"] = float(frame.execution_mfe_return_12h.mean() * 1e4)
    row["mean_deployed_gross_bps"] = float(frame.execution_gross_ev_12h.mean() * 1e4)
    row["mean_explicit_row_cost_bps"] = float(frame.execution_cost_return.mean() * 1e4)
    row["mean_deployed_net_bps"] = float(frame.execution_net_ev_12h.mean() * 1e4)
    return row


def tail_rows(frame: pd.DataFrame, layer: str, period: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    score = LAYERS[layer]["score"]
    opportunity_total = int(frame.target_economic_opportunity_hard.astype(bool).sum())
    rows: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    for fraction in TOP_FRACTIONS:
        selected = stable_top(frame, score, fraction)
        net = selected.execution_net_ev_12h.astype(float)
        gross = selected.execution_gross_ev_12h.astype(float)
        cost = selected.execution_cost_return.astype(float)
        positive = net.gt(0)
        adverse = net.le(0)
        opportunity = selected.target_economic_opportunity_hard.astype(bool)
        cutoff = float(selected[score].iloc[-1])
        ties = np.isclose(frame[score].to_numpy(float), cutoff, rtol=0.0, atol=1e-14)
        prediction = selected[score].to_numpy(float)
        calibration = bool(LAYERS[layer]["calibration_supported"])
        row = {
            "layer": layer,
            "score": score,
            "period": period,
            "top_fraction": float(fraction),
            "candidate_rows": int(len(frame)),
            "selected_rows": int(len(selected)),
            "selected_identity_sha256": selected_hash(selected),
            "selection": "one pooled-global score book; stable candidate identity ties",
            "cutoff_score": cutoff,
            "cutoff_tie_rows": int(ties.sum()),
            "full_rank_ic_net": rank_ic(frame[score], frame.execution_net_ev_12h),
            "tail_rank_ic_net": rank_ic(selected[score], net),
            "opportunity_precision": float(opportunity.mean()),
            "opportunity_recall": float(opportunity.sum() / opportunity_total) if opportunity_total else np.nan,
            "positive_net_precision": float(positive.mean()),
            "conditional_positive_net_magnitude_bps": float(selected.loc[positive, "target_positive_net_magnitude"].mean() * 1e4) if positive.any() else np.nan,
            "conditional_adverse_net_loss_bps": float(selected.loc[adverse, "target_adverse_loss_magnitude"].mean() * 1e4) if adverse.any() else np.nan,
            "full_stop_rate": float(selected.exit_is_full_stop.astype(bool).mean()),
            "timeout_rate": float(selected.exit_is_timeout.astype(bool).mean()),
            "mean_mfe_bps": float(selected.execution_mfe_return_12h.mean() * 1e4),
            "mean_deployed_gross_bps": float(gross.mean() * 1e4),
            "mean_explicit_row_cost_bps": float(cost.mean() * 1e4),
            "mean_deployed_net_bps": float(net.mean() * 1e4),
            "zero_cost_hurdle_net_bps": float(gross.mean() * 1e4),
            "fixed_100bps_hurdle_net_bps": float((gross - FIXED_COST_RETURN).mean() * 1e4),
            "observed_minus_fixed_100bps_bps": float((net - (gross - FIXED_COST_RETURN)).mean() * 1e4),
            "calibration_supported": calibration,
            "tail_prediction_bias_bps": float((prediction - net.to_numpy()).mean() * 1e4) if calibration else np.nan,
            "tail_prediction_mae_bps": float(np.abs(prediction - net.to_numpy()).mean() * 1e4) if calibration else np.nan,
        }
        rows.append(row)
        frozen = selected.loc[:, [*IDENTITY, "candidate_month", score]].copy()
        frozen["layer"] = layer
        frozen["period"] = period
        frozen["top_fraction"] = float(fraction)
        books.append(frozen)
    return rows, books


def risk_quintile(frame: pd.DataFrame) -> pd.Series:
    """Pre-entry risk-score quintiles for attribution only, not a gate."""

    score = pd.to_numeric(frame["__regime_source_execution_risk_score__"], errors="raise").to_numpy(float)
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), score))
    ranks = np.empty(len(frame), dtype=int)
    ranks[order] = np.arange(len(frame))
    return pd.Series(np.minimum((ranks * 5) // len(frame), 4), index=frame.index)


def attribution(frame: pd.DataFrame, layer: str, period: str) -> pd.DataFrame:
    score = LAYERS[layer]["score"]
    result: list[pd.DataFrame] = []
    work = frame.copy()
    work["preentry_risk_score_quintile"] = risk_quintile(work)
    for fraction in TOP_FRACTIONS:
        selected = stable_top(work, score, fraction)
        for kind, column in (
            ("side", "side_name"),
            ("asset", "__symbol__"),
            ("exit_mix", "execution_exit_class"),
            ("preentry_risk_score_quintile", "preentry_risk_score_quintile"),
        ):
            grouped = selected.groupby(column, observed=True).agg(
                selected_rows=("candidate_id", "size"),
                mean_net_bps=("execution_net_ev_12h", lambda x: float(x.mean() * 1e4)),
                positive_net_rate=("target_net_positive", "mean"),
                economic_opportunity_rate=("target_economic_opportunity_hard", "mean"),
                full_stop_rate=("exit_is_full_stop", "mean"),
                timeout_rate=("exit_is_timeout", "mean"),
            ).reset_index().rename(columns={column: "bucket"})
            grouped["layer"] = layer
            grouped["score"] = score
            grouped["period"] = period
            grouped["top_fraction"] = float(fraction)
            grouped["attribution_kind"] = kind
            grouped["selected_share"] = grouped.selected_rows / len(selected)
            result.append(grouped)
    return pd.concat(result, ignore_index=True)


def frozen_numeric_bands(frame: pd.DataFrame, layer: str) -> pd.DataFrame:
    """March numeric score bands are frozen and applied unchanged to April."""

    score = LAYERS[layer]["score"]
    months = sorted(frame.candidate_month.astype(str).unique())
    if months != ["2025-03", "2025-04"]:
        return pd.DataFrame()
    source = frame.loc[frame.candidate_month.astype(str).eq("2025-03")].copy()
    target = frame.loc[frame.candidate_month.astype(str).eq("2025-04")].copy()
    edges = np.quantile(source[score].to_numpy(float), np.linspace(0.0, 1.0, N_BANDS + 1), method="linear")
    rows: list[dict[str, Any]] = []
    for role, local in (("source_definition", source), ("target_application_of_frozen_source_cutoffs", target)):
        values = local[score].to_numpy(float)
        bands = np.searchsorted(edges[1:-1], values, side="right")
        work = local.assign(_fixed_numeric_band=bands)
        for band, cell in work.groupby("_fixed_numeric_band", sort=True, observed=True):
            lower = float(edges[int(band)])
            upper = float(edges[int(band) + 1])
            rows.append({
                "layer": layer,
                "score": score,
                "source_month": "2025-03",
                "evaluation_month": str(local.candidate_month.iloc[0]),
                "role": role,
                "band": int(band),
                "lower_inclusive_score": lower,
                "upper_score": upper,
                "upper_inclusive": int(band) == N_BANDS - 1,
                "rows": int(len(cell)),
                "population_share": float(len(cell) / len(local)),
                "mean_score": float(cell[score].mean()),
                "mean_mfe_bps": float(cell.execution_mfe_return_12h.mean() * 1e4),
                "mean_gross_bps": float(cell.execution_gross_ev_12h.mean() * 1e4),
                "mean_explicit_cost_bps": float(cell.execution_cost_return.mean() * 1e4),
                "mean_net_bps": float(cell.execution_net_ev_12h.mean() * 1e4),
                "economic_opportunity_rate": float(cell.target_economic_opportunity_hard.mean()),
                "full_stop_rate": float(cell.exit_is_full_stop.mean()),
                "timeout_rate": float(cell.exit_is_timeout.mean()),
            })
    return pd.DataFrame(rows)


def score_scale_and_cutoff_migration(frame: pd.DataFrame, layer: str) -> pd.DataFrame:
    """Label-free March->April score-scale and fixed-fraction cutoff movement."""

    score = LAYERS[layer]["score"]
    months = sorted(frame.candidate_month.astype(str).unique())
    if months != ["2025-03", "2025-04"]:
        return pd.DataFrame()
    source = frame.loc[frame.candidate_month.astype(str).eq("2025-03")]
    target = frame.loc[frame.candidate_month.astype(str).eq("2025-04")]
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        source_cutoff = float(stable_top(source, score, fraction)[score].iloc[-1])
        target_cutoff = float(stable_top(target, score, fraction)[score].iloc[-1])
        rows.append({
            "layer": layer,
            "score": score,
            "from_month": "2025-03",
            "to_month": "2025-04",
            "top_fraction": float(fraction),
            "source_rows": int(len(source)),
            "target_rows": int(len(target)),
            "source_score_mean": float(source[score].mean()),
            "target_score_mean": float(target[score].mean()),
            "score_mean_delta": float(target[score].mean() - source[score].mean()),
            "source_score_std": float(source[score].std(ddof=0)),
            "target_score_std": float(target[score].std(ddof=0)),
            "score_std_ratio": float(target[score].std(ddof=0) / source[score].std(ddof=0)) if source[score].std(ddof=0) else np.nan,
            "source_global_top_cutoff": source_cutoff,
            "target_global_top_cutoff": target_cutoff,
            "cutoff_delta": target_cutoff - source_cutoff,
            "cutoff_definition": "each period's pooled-global fixed-fraction score cutoff; descriptive only",
        })
    return pd.DataFrame(rows)


def verify_input(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path, seal_path, panel_path = root / "manifest.json", root / "manifest.sha256", root / "panel.parquet"
    if not all(path.is_file() for path in (manifest_path, seal_path, panel_path)):
        raise DiagnosticError("sealed v2 reliability input is incomplete")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise DiagnosticError("v2 reliability manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "canonical_execution_reliability_input_v2":
        raise DiagnosticError("canonical_execution_reliability_input_v2 is required")
    if manifest.get("outputs_sha256", {}).get("panel.parquet") != sha256(panel_path):
        raise DiagnosticError("v2 reliability panel hash mismatch")
    panel = pd.read_parquet(panel_path, columns=sorted(REQUIRED))
    missing = REQUIRED.difference(panel.columns)
    if missing:
        raise DiagnosticError(f"v2 reliability panel lacks required columns: {sorted(missing)}")
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True, errors="raise")
    if len(panel) != int(manifest.get("rows", -1)) or panel.duplicated(["candidate_id", "side_name"]).any():
        raise DiagnosticError("v2 reliability identity contract failed")
    if not np.allclose(panel.execution_gross_ev_12h - panel.execution_cost_return, panel.execution_net_ev_12h, rtol=0.0, atol=1e-7):
        raise DiagnosticError("gross - explicit row cost != deployed net")
    if sorted(panel.candidate_month.astype(str).unique()) != ["2025-03", "2025-04"]:
        raise DiagnosticError("expected strict March-April 2025 reliability population")
    causal = panel.causal_pooled_21d_eligible.eq(True)
    if causal.any() and panel.loc[causal, "causal_pooled_21d"].isna().any():
        raise DiagnosticError("causal mapped eligible rows lack a mapped score")
    return panel, manifest


def run(*, input_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    panel, input_manifest = verify_input(input_root)
    bridge_rows: list[dict[str, Any]] = []
    tail_metrics: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    attribution_parts: list[pd.DataFrame] = []
    bands: list[pd.DataFrame] = []
    score_migration: list[pd.DataFrame] = []
    layer_coverage: list[dict[str, Any]] = []
    for layer in LAYERS:
        eligible = layer_rows(panel, layer)
        layer_coverage.append({"layer": layer, "score": LAYERS[layer]["score"], "rows": int(len(eligible)), "coverage_fraction": float(len(eligible) / len(panel)), "months": ",".join(sorted(eligible.candidate_month.astype(str).unique())), "calibration_supported": bool(LAYERS[layer]["calibration_supported"])})
        for period, local in periods(eligible):
            bridge_rows.append(bridge(local, layer, period))
            tail, frozen = tail_rows(local, layer, period)
            tail_metrics.extend(tail)
            books.extend(frozen)
            attribution_parts.append(attribution(local, layer, period))
        fixed = frozen_numeric_bands(eligible, layer)
        if not fixed.empty:
            bands.append(fixed)
        migration = score_scale_and_cutoff_migration(eligible, layer)
        if not migration.empty:
            score_migration.append(migration)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        outputs: Mapping[str, pd.DataFrame] = {
            "layer_coverage.csv": pd.DataFrame(layer_coverage),
            "identical_cohort_ic_bridge.csv": pd.DataFrame(bridge_rows),
            "global_tail_metrics.csv": pd.DataFrame(tail_metrics),
            "frozen_selected_books.parquet": pd.concat(books, ignore_index=True),
            "tail_attribution.csv": pd.concat(attribution_parts, ignore_index=True),
            "frozen_numeric_band_response.csv": pd.concat(bands, ignore_index=True) if bands else pd.DataFrame(),
            "score_scale_cutoff_migration.csv": pd.concat(score_migration, ignore_index=True) if score_migration else pd.DataFrame(),
        }
        for name, value in outputs.items():
            if name.endswith(".parquet"):
                value.to_parquet(stage / name, index=False, compression="zstd")
            else:
                value.to_csv(stage / name, index=False)
        report = {
            "schema": "canonical_execution_reliability_ic_ev_diagnostic_v1",
            "status": "DIAGNOSTIC_ONLY_NO_MODEL_FIT_NO_MAPPING_FIT_NO_PROMOTION_NO_REPLAY",
            "promotion_eligible": False,
            "input": {"root": str(input_root), "manifest_sha256": sha256(input_root / "manifest.json"), "panel_sha256": sha256(input_root / "panel.parquet"), "rows": int(len(panel))},
            "contracts": {
                "population": "exact sealed v2 candidate_id+side_name cohort; UTC timestamp parity inherited from v2",
                "selection": "one pooled-global cross-side/cross-timestamp top 1/5/10/20 score book within each period; side, asset, exit and pre-entry-risk rows are attribution only",
                "waterfall": "the same frozen selected identities are used for MFE, deployed gross, observed row cost, deployed net, zero-cost gross and fixed 100bps counterfactual; no stage reselects",
                "native_vs_execution": "native alpha target is 24h while execution targets are exact 12h; this target-horizon mismatch is reported rather than treated as calibration",
                "mapping": "only causal_pooled_21d eligible April rows are called causally mapped; it maps raw_score, not direct_q25_return",
                "frozen_numeric_bands": "March pooled score decile edges are labels-free score thresholds frozen and applied unchanged to April; they are response diagnostics, not admission thresholds",
                "fixed_candidate_cohort": "frozen_selected_books.parquet is the exact selected identity ledger for every layer/period/top-k; cross-month candidate IDs are not falsely matched",
                "cost": "execution_gross_ev_12h - execution_cost_return == execution_net_ev_12h; fixed 100bps and zero-cost values are counterfactuals on that same book only",
                "regime": "pre-entry execution-risk-score quintiles are attribution-only deterministic ranks, not learned or selected regimes",
            },
            "limitations": [
                "The sealed v2 population begins in March, so this artifact cannot reconcile the February-to-March shift on an identical March-April cohort.",
                "No raw score is relabelled or newly calibrated; native alpha, residual, q25 and causal raw-score map semantics remain separate.",
                "This is not evidence for a policy or portfolio promotion.",
            ],
            "outputs_sha256": {name: sha256(stage / name) for name in outputs},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        write_json(stage / "manifest.json", report)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "  manifest.json\n")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--input-root", type=Path, default=INPUT)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(input_root=args.input_root, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
