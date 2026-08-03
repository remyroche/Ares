#!/usr/bin/env python3
"""Repair only the cross-side common-unit map of frozen gross-cost hurdle scores.

No execution model is fitted here.  The source's `gross_cost_hurdle_ev` scores
and exact labels are immutable.  Each map is refit online per UTC day from
only earlier, resolved rows in a 21-day window; the selected configuration is
chosen on pre-June OOF only and frozen before either forward evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/hurdle_cross_side_common_unit_mapping_20260730_v1"
SCHEMA = "hurdle_cross_side_common_unit_mapping_v1"
WINDOW_DAYS = 21
MIN_GLOBAL_ROWS = 500
MIN_SIDE_ROWS = 100
SHRINK_GRID = (250.0, 1_000.0, 4_000.0)
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
SIDES = ("long", "short")
SCORE = "gross_cost_hurdle_ev"
CANONICAL = "canonical_recent_ev_score_gross_cost_hurdle_ev"
REQUIRED = (
    "candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc",
    "support_label_available_utc", "execution_net_ev_12h", SCORE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path, *, oof: bool) -> pd.DataFrame:
    columns = [*REQUIRED, *( ["oof_fold"] if oof else []), "window"]
    if not oof:
        columns.append(CANONICAL)
    frame = pd.read_parquet(path, columns=columns).copy()
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"frozen source lacks {missing}")
    if frame.duplicated(["candidate_id", "window"]).any():
        raise ValueError("source candidate/window identity is not unique")
    for column in ("__ts__", "execution_decision_utc", "support_label_available_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    for column in (SCORE, "execution_net_ev_12h"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame["execution_net_ev_12h"].to_numpy(float)).all():
        raise ValueError("frozen exact net target has non-finite values")
    if not oof and not np.isfinite(frame[SCORE].to_numpy(float)).all():
        raise ValueError("frozen forward score has non-finite values")
    if not frame.side_name.astype(str).isin(SIDES).all():
        raise ValueError("unexpected side")
    return frame


def _fit(reference: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(reference[SCORE].to_numpy(float), reference.execution_net_ev_12h.to_numpy(float))
    return model.predict(values)


def causal_map(reference_source: pd.DataFrame, target: pd.DataFrame, *, shrink: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each target day using only strictly earlier resolved 21d evidence.

    `shrink=None` means a pooled-only common anchor.  A weak global reference
    is exactly zero—not raw score—so cross-side units never silently revert to
    incomparable model scales.  Weak side support leaves the side residual at
    exactly zero while retaining the pooled anchor.
    """

    source = reference_source.loc[:, [*REQUIRED]].copy()
    source["origin"] = "reference"
    current = target.loc[:, [*REQUIRED]].copy()
    current["origin"] = "target"
    universe = pd.concat([source, current], ignore_index=True).sort_values(["__ts__", "candidate_id", "origin"], kind="stable").drop_duplicates(["candidate_id"], keep="first").reset_index(drop=True)
    result = target.copy().reset_index(drop=True)
    result["mapped_score"] = np.nan
    result["pooled_anchor"] = np.nan
    result["side_residual"] = 0.0
    result["side_shrink_weight"] = 0.0
    result["map_status"] = "unmapped"
    audits: list[dict[str, Any]] = []
    for snapshot, local_index in result.groupby(result.__ts__.dt.floor("D"), sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        reference = universe.loc[
            universe.__ts__.lt(snapshot)
            & universe.__ts__.ge(snapshot - pd.Timedelta(days=WINDOW_DAYS))
            & universe.support_label_available_utc.lt(snapshot)
        ].copy()
        target_positions = np.asarray(list(local_index), dtype=int)
        raw = result.loc[target_positions, SCORE].to_numpy(float)
        if len(reference) < MIN_GLOBAL_ROWS or reference[SCORE].nunique() < 2:
            # Exact zero fallback deliberately has a single, globally shared
            # unit. It is not a raw-score fallback and cannot rank by side.
            result.loc[target_positions, ["mapped_score", "pooled_anchor", "side_residual", "side_shrink_weight", "map_status"]] = [0.0, 0.0, 0.0, 0.0, "zero_fallback_weak_global"]
            audits.append({"snapshot_utc": snapshot, "rows": len(target_positions), "reference_rows": len(reference), "status": "zero_fallback_weak_global", "long_reference_rows": int(reference.side_name.eq("long").sum()), "short_reference_rows": int(reference.side_name.eq("short").sum())})
            continue
        pooled = _fit(reference, raw)
        result.loc[target_positions, "pooled_anchor"] = pooled
        result.loc[target_positions, "mapped_score"] = pooled
        result.loc[target_positions, "map_status"] = "pooled_anchor"
        side_counts: dict[str, int] = {}
        side_weights: dict[str, float] = {}
        if shrink is not None:
            for side in SIDES:
                mask = result.loc[target_positions, "side_name"].astype(str).eq(side).to_numpy()
                side_ref = reference.loc[reference.side_name.astype(str).eq(side)]
                side_counts[side] = int(len(side_ref))
                if not mask.any() or len(side_ref) < MIN_SIDE_ROWS or side_ref[SCORE].nunique() < 2:
                    side_weights[side] = 0.0
                    continue
                side_values = _fit(side_ref, raw[mask])
                weight = float(len(side_ref) / (len(side_ref) + float(shrink)))
                positions = target_positions[mask]
                residual = weight * (side_values - pooled[mask])
                result.loc[positions, "side_residual"] = residual
                result.loc[positions, "side_shrink_weight"] = weight
                result.loc[positions, "mapped_score"] = pooled[mask] + residual
                result.loc[positions, "map_status"] = "pooled_plus_shrunk_side_residual"
                side_weights[side] = weight
        audits.append({"snapshot_utc": snapshot, "rows": len(target_positions), "reference_rows": len(reference), "status": "pooled" if shrink is None else "pooled_plus_side_residual", "long_reference_rows": int(reference.side_name.eq("long").sum()), "short_reference_rows": int(reference.side_name.eq("short").sum()), "long_side_shrink_weight": side_weights.get("long", 0.0), "short_side_shrink_weight": side_weights.get("short", 0.0), "mean_abs_side_residual_bps": float(np.abs(result.loc[target_positions, "side_residual"]).mean() * 1e4)})
    if result.mapped_score.isna().any() or not np.isfinite(result.mapped_score).all():
        raise AssertionError("causal map did not yield a finite score for every target row")
    return result, pd.DataFrame(audits)


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), -frame[score].to_numpy(float)))
    return frame.iloc[order[:count]].copy()


def _calibration(frame: pd.DataFrame, score: str) -> dict[str, float]:
    x, y = frame[score].to_numpy(float), frame.execution_net_ev_12h.to_numpy(float)
    if np.unique(x).size < 2:
        return {"calibration_slope": np.nan, "calibration_intercept_bps": np.nan, "calibration_mae_bps": float(np.abs(y - x).mean() * 1e4)}
    slope, intercept = np.polyfit(x, y, 1)
    return {"calibration_slope": float(slope), "calibration_intercept_bps": float(intercept * 1e4), "calibration_mae_bps": float(np.abs(y - x).mean() * 1e4)}


def _tie_metrics(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, Any]:
    values = frame[score].to_numpy(float); target = frame.execution_net_ev_12h.to_numpy(float)
    count = max(1, int(math.ceil(len(frame) * fraction))); cutoff = float(np.sort(values)[-count])
    above, plateau = values > cutoff, values == cutoff
    need = count - int(above.sum()); p = target[plateau]
    expected_sum = float(target[above].sum() + need * p.mean())
    lower_sum = float(target[above].sum() + np.sort(p)[:need].sum())
    upper_sum = float(target[above].sum() + np.sort(p)[-need:].sum())
    return {"score_unique_count": int(np.unique(values).size), "cutoff": cutoff, "cutoff_plateau_rows": int(plateau.sum()), "selected_from_cutoff_plateau": int(need), "cutoff_tie_ambiguous": bool(int(plateau.sum()) > need), "tie_expected_net_bps": expected_sum / count * 1e4, "tie_lower_net_bps": lower_sum / count * 1e4, "tie_upper_net_bps": upper_sum / count * 1e4}


def evaluate(frame: pd.DataFrame, *, score: str, map_arm: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []; sides: list[dict[str, Any]] = []; assets: list[dict[str, Any]] = []
    week = frame.__ts__.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    scopes = [("all", frame), ("latest_week", frame.loc[week.eq(week.max())])]
    for scope, local in scopes:
        for fraction in FRACTIONS:
            selected = stable_top(local, score, fraction)
            weights = selected.__symbol__.value_counts(normalize=True)
            row = {"window": window, "map_arm": map_arm, "scope": scope, "fraction": fraction, "candidate_rows": len(local), "selected_rows": len(selected), "mean_net_bps": float(selected.execution_net_ev_12h.mean() * 1e4), "positive_net_rate": float(selected.execution_net_ev_12h.gt(0).mean()), "side_long_share": float(selected.side_name.eq("long").mean()), "side_short_share": float(selected.side_name.eq("short").mean()), "asset_count": int(selected.__symbol__.nunique()), "asset_top_share": float(weights.iloc[0]), "asset_hhi": float((weights ** 2).sum()), "zero_fallback_rate": float(selected.map_status.eq("zero_fallback_weak_global").mean()) if "map_status" in selected else np.nan, **_tie_metrics(local, score, fraction), **_calibration(local, score)}
            metrics.append(row)
            for side, part in selected.groupby("side_name", sort=True):
                sides.append({"window": window, "map_arm": map_arm, "scope": scope, "fraction": fraction, "side_name": side, "selected_rows": len(part), "selected_share": len(part) / len(selected), "mean_net_bps": float(part.execution_net_ev_12h.mean() * 1e4), "positive_net_rate": float(part.execution_net_ev_12h.gt(0).mean())})
            for asset, part in selected.groupby("__symbol__", sort=True):
                assets.append({"window": window, "map_arm": map_arm, "scope": scope, "fraction": fraction, "__symbol__": asset, "selected_rows": len(part), "selected_share": len(part) / len(selected), "mean_net_bps": float(part.execution_net_ev_12h.mean() * 1e4)})
    return pd.DataFrame(metrics), pd.DataFrame(sides), pd.DataFrame(assets)


def choose_shrink(oof: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Choose a frozen shrink setting using only pre-June resolved OOF outcomes."""
    rows: list[dict[str, Any]] = []
    for shrink in SHRINK_GRID:
        mapped, _ = causal_map(oof, oof, shrink=shrink)
        evaluation, _, _ = evaluate(mapped, score="mapped_score", map_arm=f"side_residual_shrink_{int(shrink)}", window="pre_june_oof")
        top10 = evaluation.query("scope == 'all' and fraction == .1").iloc[0]
        rows.append({"shrink": shrink, "top10_net_bps": float(top10.mean_net_bps), "top10_positive_rate": float(top10.positive_net_rate), "top10_tie_ambiguous": bool(top10.cutoff_tie_ambiguous), "coverage_rate": float(np.isfinite(mapped.mapped_score).mean()), "zero_fallback_rate": float(mapped.map_status.eq("zero_fallback_weak_global").mean())})
    ledger = pd.DataFrame(rows).sort_values(["coverage_rate", "top10_net_bps", "top10_positive_rate", "shrink"], ascending=[False, False, False, True], kind="stable").reset_index(drop=True)
    return float(ledger.iloc[0].shrink), ledger


def run(*, source_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = {"manifest": source_root / "manifest.json", "seal": source_root / "manifest.sha256", "oof": source_root / "support_head_oof_ledger.parquet", "forward": source_root / "forward_predictions.parquet"}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("frozen hurdle source is incomplete")
    if paths["seal"].read_text().split()[0] != sha256(paths["manifest"]):
        raise ValueError("frozen hurdle manifest checksum fails")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema") != "exact_strict_oof_hurdle_distributional_ablation_v1" or manifest.get("outputs", {}).get("forward_predictions", {}).get("sha256") != sha256(paths["forward"]):
        raise ValueError("wrong or changed frozen hurdle source")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    oof = _load(paths["oof"], oof=True).loc[lambda x: x[SCORE].notna()].copy()
    forward = _load(paths["forward"], oof=False)
    # Config selection uses only the May OOF control resolved before the first
    # forward cutoff, then remains fixed for both May->June and later-July.
    freeze = pd.Timestamp("2026-06-01T00:00:00Z")
    selection_oof = oof.loc[(oof.window.eq("may_to_june_forward_control")) & oof.support_label_available_utc.lt(freeze)].copy()
    selected_shrink, selection_ledger = choose_shrink(selection_oof)
    all_metrics: list[pd.DataFrame] = []; all_sides: list[pd.DataFrame] = []; all_assets: list[pd.DataFrame] = []; all_candidates: list[pd.DataFrame] = []; all_audits: list[pd.DataFrame] = []
    for window, target in forward.groupby("window", sort=True):
        reference = oof.loc[oof.window.eq(window)].copy()
        variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {"canonical_existing": (target.assign(mapped_score=target[CANONICAL], map_status="source_canonical"), pd.DataFrame())}
        variants["pooled_global_anchor"] = causal_map(reference, target, shrink=None)
        for shrink in SHRINK_GRID:
            variants[f"pooled_plus_side_residual_shrink_{int(shrink)}"] = causal_map(reference, target, shrink=shrink)
        for arm, (mapped, audit) in variants.items():
            metric, sides, assets = evaluate(mapped, score="mapped_score", map_arm=arm, window=str(window))
            all_metrics.append(metric); all_sides.append(sides); all_assets.append(assets)
            keep = ["candidate_id", "__ts__", "__symbol__", "side_name", "execution_net_ev_12h", SCORE, CANONICAL, "mapped_score", "pooled_anchor", "side_residual", "side_shrink_weight", "map_status"]
            for missing in set(keep).difference(mapped.columns): mapped[missing] = np.nan
            all_candidates.append(mapped.loc[:, keep].assign(window=window, map_arm=arm))
            if not audit.empty: all_audits.append(audit.assign(window=window, map_arm=arm))
    metrics = pd.concat(all_metrics, ignore_index=True); sides = pd.concat(all_sides, ignore_index=True); assets = pd.concat(all_assets, ignore_index=True); candidates = pd.concat(all_candidates, ignore_index=True); audit = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()
    selected_arm = f"pooled_plus_side_residual_shrink_{int(selected_shrink)}"
    top10 = metrics.query("map_arm == @selected_arm and scope == 'all' and fraction == .1")
    gates = {"selected_configuration_frozen_before_forward": True, "all_forward_top10_positive": bool((top10.mean_net_bps > 0).all()), "all_forward_latest_week_positive": bool((metrics.query("map_arm == @selected_arm and scope == 'latest_week' and fraction == .1").mean_net_bps > 0).all()), "all_forward_tie_unambiguous": bool(~top10.cutoff_tie_ambiguous.astype(bool).any()), "no_side_quota": True, "portfolio_replay": "NOT_RUN"}
    output_dir.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    outputs = {"selection_ledger.csv": selection_ledger, "forward_metrics.csv": metrics, "side_attribution.csv": sides, "asset_concentration.csv": assets, "mapped_candidates.parquet": candidates, "daily_mapping_audit.csv": audit}
    for name, table in outputs.items():
        path = temporary / name
        table.to_parquet(path, index=False, compression="zstd") if name.endswith(".parquet") else table.to_csv(path, index=False)
    report = {"schema": SCHEMA, "status": "FROZEN_HURDLE_COMMON_UNIT_MAPPING_DIAGNOSTIC_COMPLETE", "promotion_eligible": False, "selected_shrink": selected_shrink, "selected_forward_arm": selected_arm, "freeze": {"selection_source": "May OOF only", "selection_cutoff_utc": freeze, "config_applied_to": sorted(forward.window.unique())}, "contracts": {"no_refit": "only frozen gross_cost_hurdle_ev predictions are read", "daily_causal_state": "UTC daily 21d reference; row must be strictly earlier and support label resolved before snapshot", "common_unit": "pooled isotonic net-EV anchor, then optional side residual in net units shrunk toward exactly zero", "weak_support": "global weak support maps to exact 0.0; weak side support contributes exactly zero residual", "selection": "one pooled-global top1/5/10/20 with deterministic candidate-ID ties; no timestamp/side/asset quota", "portfolio": "not replayed; explicit gates are diagnostic only"}, "gates": gates, "source": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items() if name != "seal"}, "outputs_sha256": {name: sha256(temporary / name) for name in outputs}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
    write_json(temporary / "manifest.json", report); (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n", encoding="utf-8"); os.replace(temporary, output_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps(safe(run(source_root=args.source_root, output_dir=args.output_dir)), sort_keys=True))
