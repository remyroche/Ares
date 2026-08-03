#!/usr/bin/env python3
"""Materialize exact decision-time path targets for a separate action layer.

The pack contains outcomes only.  It is never an inference feature source and
does not change the execution-EV ranking.  Each horizon has an explicit label
availability timestamp, and canonical round-trip cost is deducted exactly once
from fixed-close counterfactuals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v4"
PATH_ROOT = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1"
OUT = ROOT / "data_perp/artifacts/execution_action_target_pack_20260730_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
HORIZONS = (1, 2, 3, 4, 8, 12)
FIXED_ACTION_HORIZONS = (1, 2, 4, 8, 12)
BUFFERS_BPS = (0, 25, 50)


class ContractError(RuntimeError):
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ContractError(f"missing seal: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ContractError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ContractError(f"schema mismatch: {root}")
    for name, expected in manifest.get("outputs_sha256", {}).items():
        if sha256(root / name) != expected:
            raise ContractError(f"sealed output mismatch: {root / name}")
    return manifest


def _path_arrays(payload: Any) -> tuple[np.ndarray, ...]:
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    timestamp = np.asarray(parsed["timestamp"], dtype=np.int64)
    arrays = tuple(
        np.asarray(parsed[name], dtype=np.float64)
        for name in ("open", "high", "low", "close")
    )
    if (
        timestamp.shape != (720,)
        or any(array.shape != (720,) for array in arrays)
        or any(not np.isfinite(array).all() for array in arrays)
        or not np.all(np.diff(timestamp) == 60_000_000_000)
    ):
        raise ContractError("path must contain 720 contiguous finite 1-minute bars")
    return timestamp, *arrays


def compute_action_targets(
    payload: Any,
    *,
    decision_price: float,
    side_name: str,
    cost_return: float,
    atr_1h: float,
) -> dict[str, Any]:
    """Compute target-only side-relative path geometry from decision onward."""
    timestamp, open_, high, low, close = _path_arrays(payload)
    if not np.isfinite(decision_price) or decision_price <= 0:
        raise ContractError("invalid decision price")
    if not np.isfinite(cost_return) or cost_return < 0:
        raise ContractError("invalid canonical cost")
    atr_fraction = float(atr_1h) / float(decision_price)
    if not np.isfinite(atr_fraction) or atr_fraction <= 0:
        raise ContractError("invalid decision-time ATR")
    side = str(side_name).lower()
    if side == "long":
        close_return = close / decision_price - 1.0
        favorable = high / decision_price - 1.0
        adverse = 1.0 - low / decision_price
    elif side == "short":
        close_return = 1.0 - close / decision_price
        favorable = 1.0 - low / decision_price
        adverse = high / decision_price - 1.0
    else:
        raise ContractError(f"unknown side: {side_name}")
    favorable = np.maximum(favorable, 0.0)
    adverse = np.maximum(adverse, 0.0)
    result: dict[str, Any] = {
        "path_start_utc": pd.to_datetime(timestamp[0], unit="ns", utc=True),
        "path_last_minute_utc": pd.to_datetime(timestamp[-1], unit="ns", utc=True),
        "target_side_sign": 1 if side == "long" else -1,
        "target_atr_fraction": atr_fraction,
    }
    for hours in HORIZONS:
        count = hours * 60
        fixed_gross = float(close_return[count - 1])
        mfe = float(np.max(favorable[:count]))
        mae = float(np.max(adverse[:count]))
        underwater = np.maximum(-close_return[:count], 0.0)
        result.update(
            {
                f"target_fixed_{hours}h_gross_return": fixed_gross,
                f"target_fixed_{hours}h_net_return": fixed_gross - cost_return,
                f"target_mfe_{hours}h_return": mfe,
                f"target_mae_{hours}h_return": mae,
                f"target_mfe_{hours}h_atr": mfe / atr_fraction,
                f"target_mae_{hours}h_atr": mae / atr_fraction,
                f"target_underwater_fraction_{hours}h": float(
                    np.mean(close_return[:count] < 0.0)
                ),
                f"target_underwater_area_atr_hours_{hours}h": float(
                    underwater.sum() / 60.0 / atr_fraction
                ),
                f"target_path_slope_atr_per_hour_{hours}h": float(
                    fixed_gross / atr_fraction / hours
                ),
            }
        )
    peak = float(np.max(favorable))
    peak_valid = peak > 0.0
    peak_index = int(np.argmax(favorable)) if peak_valid else 719
    result["target_peak_mfe_12h_return"] = peak
    result["target_peak_mfe_timing_valid_12h"] = int(peak_valid)
    result["target_time_to_peak_mfe_minutes_12h"] = (
        peak_index + 1 if peak_valid else np.nan
    )
    result["target_time_to_peak_mfe_hours_12h"] = (
        (peak_index + 1) / 60.0 if peak_valid else np.nan
    )
    result["target_time_to_peak_mfe_censored_hours_12h"] = (
        (peak_index + 1) / 60.0 if peak_valid else 12.0
    )
    for fraction in (0.50, 0.80):
        label = int(fraction * 100)
        threshold = fraction * peak
        index = (
            int(np.flatnonzero(favorable >= threshold - 1e-15)[0])
            if peak_valid
            else 719
        )
        result[f"target_time_to_{label}pct_mfe_minutes_12h"] = (
            index + 1 if peak_valid else np.nan
        )
        result[f"target_time_to_{label}pct_mfe_hours_12h"] = (
            (index + 1) / 60.0 if peak_valid else np.nan
        )
        result[f"target_time_to_{label}pct_mfe_censored_hours_12h"] = (
            (index + 1) / 60.0 if peak_valid else 12.0
        )
        if label == 80:
            result["target_underwater_minutes_before_80pct_mfe"] = int(
                np.sum(close_return[: index + 1] < 0.0)
            )
            result["target_underwater_fraction_before_80pct_mfe"] = float(
                np.mean(close_return[: index + 1] < 0.0)
            )
            running_peak = np.maximum.accumulate(close_return[index:])
            giveback = running_peak - close_return[index:]
            result["target_max_close_giveback_after_80pct_mfe_return"] = (
                float(np.max(giveback)) if peak_valid else np.nan
            )
            result["target_max_close_giveback_after_80pct_mfe_ratio"] = (
                float(np.max(giveback) / peak) if peak_valid else np.nan
            )
    result["target_final_close_giveback_from_peak_return"] = (
        peak - float(close_return[-1]) if peak_valid else np.nan
    )
    result["target_final_close_giveback_from_peak_ratio"] = (
        (peak - float(close_return[-1])) / peak if peak_valid else np.nan
    )
    post_peak_close = close_return[peak_index:]
    result["target_worst_post_peak_close_giveback_return"] = (
        float(peak - np.min(post_peak_close)) if peak_valid else np.nan
    )
    result["target_worst_post_peak_close_giveback_ratio"] = (
        (peak - np.min(post_peak_close)) / peak if peak_valid else np.nan
    )
    for buffer_bps in BUFFERS_BPS:
        hurdle = cost_return + buffer_bps / 10_000.0
        hit = favorable >= hurdle
        occurred = bool(np.any(hit))
        label = f"{buffer_bps}bps"
        result[f"target_cost_clear_opportunity_{label}"] = int(occurred)
        result[f"target_time_to_cost_clear_{label}_minutes"] = (
            int(np.argmax(hit)) + 1 if occurred else np.nan
        )
        result[f"target_time_to_cost_clear_{label}_hours"] = (
            (int(np.argmax(hit)) + 1) / 60.0 if occurred else np.nan
        )
        result[f"target_time_to_cost_clear_{label}_censored_hours"] = (
            (int(np.argmax(hit)) + 1) / 60.0 if occurred else 12.0
        )
    for hours in (2, 3):
        close_atr = result[f"target_fixed_{hours}h_gross_return"] / atr_fraction
        mae_atr = result[f"target_mae_{hours}h_atr"]
        result[f"target_early_{hours}h_flat_close_le_0p25atr"] = int(
            abs(close_atr) <= 0.25
        )
        result[f"target_early_{hours}h_mae_ge_0p50atr"] = int(mae_atr >= 0.50)
        result[f"target_early_{hours}h_clean_nonflat"] = int(
            close_atr > 0.25 and mae_atr < 0.50
        )
        result[f"target_early_{hours}h_positive_after_cost"] = int(
            result[f"target_fixed_{hours}h_net_return"] > 0.0
        )
    fixed_values = np.asarray(
        [result[f"target_fixed_{hours}h_net_return"] for hours in FIXED_ACTION_HORIZONS]
    )
    best_index = int(np.argmax(fixed_values))
    result["target_best_fixed_horizon_hours_diagnostic_only"] = int(
        FIXED_ACTION_HORIZONS[best_index]
    )
    result["target_best_fixed_horizon_net_return_diagnostic_only"] = float(
        fixed_values[best_index]
    )
    result["target_best_fixed_horizon_gain_vs_12h_diagnostic_only"] = float(
        fixed_values[best_index] - result["target_fixed_12h_net_return"]
    )
    return result


def _support(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (month, side), local in frame.groupby(
        ["candidate_month", "side_name"], sort=True
    ):
        row: dict[str, Any] = {
            "candidate_month": month,
            "side_name": side,
            "rows": int(len(local)),
            "decision_min_utc": local.execution_decision_utc.min(),
            "decision_max_utc": local.execution_decision_utc.max(),
            "label_12h_max_utc": local.label_available_at_12h_utc.max(),
        }
        for buffer_bps in BUFFERS_BPS:
            row[f"opportunity_{buffer_bps}bps_rate"] = float(
                local[f"target_cost_clear_opportunity_{buffer_bps}bps"].mean()
            )
        for hours in (2, 3):
            row[f"early_{hours}h_clean_nonflat_rate"] = float(
                local[f"target_early_{hours}h_clean_nonflat"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(output: Path = OUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    panel_manifest = verify_seal(
        PANEL_ROOT, "canonical_execution_reliability_input_v4"
    )
    path_manifest = json.loads((PATH_ROOT / "manifest.json").read_text())
    if (
        path_manifest.get("cost_accounting")
        != "fee_once_entry_spread_once_exit_spread_once"
        or any(
            path_manifest.get("coverage", {}).get("by_month", {}).get(month, {}).get(
                "coverage"
            )
            != 1.0
            for month in ("2025-03", "2025-04")
        )
    ):
        raise ContractError("exact execution-path source is incomplete")
    panel = pd.read_parquet(
        PANEL_ROOT / "panel.parquet",
        columns=[
            *IDENTITY,
            "execution_decision_utc",
            "execution_label_end_utc",
            "candidate_month",
            "execution_cost_return",
        ],
    )
    paths = pd.read_parquet(
        PATH_ROOT / "paths.parquet",
        columns=[
            *IDENTITY,
            "execution_future_path",
            "decision_price",
            "atr_1h",
        ],
    )
    paths = paths.rename(
        columns={
            "__symbol__": "path_symbol",
            "__ts__": "path_signal_utc",
        }
    )
    joined = panel.merge(
        paths,
        on=["candidate_id", "side_name"],
        how="inner",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != len(panel) or not joined._merge.eq("both").all():
        raise ContractError("exact path source does not cover the canonical panel")
    joined = joined.drop(columns="_merge")
    for column in (
        "__ts__",
        "path_signal_utc",
        "execution_decision_utc",
        "execution_label_end_utc",
    ):
        joined[column] = pd.to_datetime(joined[column], utc=True)
    if (
        not joined.path_signal_utc.eq(joined["__ts__"]).all()
        or not joined.path_symbol.astype(str)
        .str.replace("/", "_", regex=False)
        .eq(joined["__symbol__"].astype(str))
        .all()
    ):
        raise ContractError("path signal timestamp/symbol parity failed")
    records = []
    row_columns = [
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        "execution_cost_return",
        "execution_future_path",
        "decision_price",
        "atr_1h",
    ]
    for (
        candidate_id,
        side_name,
        symbol,
        signal_utc,
        decision_utc,
        label_end_utc,
        candidate_month,
        cost_return,
        payload,
        decision_price,
        atr_1h,
    ) in joined.loc[:, row_columns].itertuples(index=False, name=None):
        target = compute_action_targets(
            payload,
            decision_price=float(decision_price),
            side_name=str(side_name),
            cost_return=float(cost_return),
            atr_1h=float(atr_1h),
        )
        record = {
            "candidate_id": candidate_id,
            "side_name": side_name,
            "__symbol__": symbol,
            "__ts__": signal_utc,
            "execution_decision_utc": decision_utc,
            "execution_label_end_utc": label_end_utc,
            "candidate_month": candidate_month,
            "canonical_cost_return": cost_return,
            **target,
        }
        records.append(record)
    labels = pd.DataFrame(records)
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        labels[column] = pd.to_datetime(labels[column], utc=True)
    for hours in HORIZONS:
        labels[f"label_available_at_{hours}h_utc"] = (
            labels.execution_decision_utc + pd.Timedelta(hours=hours)
        )
    if (
        len(labels) != 110_730
        or labels.duplicated(list(IDENTITY)).any()
        or not labels.path_start_utc.eq(labels.execution_decision_utc).all()
        or not labels.path_last_minute_utc.eq(
            labels.execution_decision_utc + pd.Timedelta(minutes=719)
        ).all()
        or not labels.execution_label_end_utc.eq(
            labels.execution_decision_utc + pd.Timedelta(hours=12)
        ).all()
    ):
        raise ContractError("target identity/horizon timing contract failed")
    for hours in HORIZONS:
        if not labels[f"label_available_at_{hours}h_utc"].eq(
            labels.execution_decision_utc + pd.Timedelta(hours=hours)
        ).all():
            raise ContractError(f"{hours}h label availability drift")
        if not np.allclose(
            labels[f"target_fixed_{hours}h_gross_return"]
            - labels.canonical_cost_return,
            labels[f"target_fixed_{hours}h_net_return"],
            atol=1e-12,
        ):
            raise ContractError(f"{hours}h cost accounting drift")
    if not (
        labels.target_cost_clear_opportunity_50bps
        .le(labels.target_cost_clear_opportunity_25bps)
        .all()
        and labels.target_cost_clear_opportunity_25bps
        .le(labels.target_cost_clear_opportunity_0bps)
        .all()
    ):
        raise ContractError("cost-buffer opportunity nesting failed")
    target_roles = {
        "schema": "execution_action_target_roles_v1",
        "status": "TARGET_ONLY_NEVER_MODEL_INPUT",
        "identity": list(IDENTITY),
        "fixed_horizon_actions": {
            str(hours): {
                "gross": f"target_fixed_{hours}h_gross_return",
                "net": f"target_fixed_{hours}h_net_return",
                "available_at": f"label_available_at_{hours}h_utc",
            }
            for hours in FIXED_ACTION_HORIZONS
        },
        "supporting_labels": {
            "timing": [
                "target_peak_mfe_timing_valid_12h",
                "target_time_to_peak_mfe_censored_hours_12h",
                "target_time_to_80pct_mfe_censored_hours_12h",
                *[
                    f"target_time_to_cost_clear_{buffer}bps_censored_hours"
                    for buffer in BUFFERS_BPS
                ],
            ],
            "early_path": [
                *[
                    f"target_early_{hours}h_clean_nonflat"
                    for hours in (2, 3)
                ],
                *[
                    f"target_early_{hours}h_mae_ge_0p50atr"
                    for hours in (2, 3)
                ],
            ],
            "underwater": [
                "target_underwater_minutes_before_80pct_mfe",
                "target_underwater_fraction_before_80pct_mfe",
                *[
                    f"target_underwater_area_atr_hours_{hours}h"
                    for hours in HORIZONS
                ],
            ],
            "giveback": [
                "target_final_close_giveback_from_peak_ratio",
                "target_worst_post_peak_close_giveback_ratio",
                "target_max_close_giveback_after_80pct_mfe_ratio",
            ],
            "slope": [
                f"target_path_slope_atr_per_hour_{hours}h"
                for hours in HORIZONS
            ],
        },
        "diagnostic_only_hindsight": [
            "target_best_fixed_horizon_hours_diagnostic_only",
            "target_best_fixed_horizon_net_return_diagnostic_only",
            "target_best_fixed_horizon_gain_vs_12h_diagnostic_only",
        ],
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        labels.to_parquet(stage / "labels.parquet", index=False, compression="zstd")
        support = _support(labels)
        support.to_csv(stage / "support_by_month_side.csv", index=False)
        write_json(stage / "target_roles.json", target_roles)
        outputs = {
            name: sha256(stage / name)
            for name in (
                "labels.parquet",
                "support_by_month_side.csv",
                "target_roles.json",
            )
        }
        manifest = {
            "schema": "execution_action_target_pack_v2",
            "status": "SEALED_TARGET_ONLY_NO_MODEL_INPUT_NO_PROMOTION_NO_POLICY_REPLAY",
            "promotion_eligible": False,
            "rows": int(len(labels)),
            "contract": {
                "path": "exact 720x1m path beginning at execution_decision_utc",
                "side": "all returns, MFE, MAE, timing, slope, underwater and giveback labels are side-relative",
                "cost": "canonical row round-trip cost deducted exactly once from every fixed-close net target",
                "availability": "each fixed/path-prefix target is usable only at its declared decision+horizon timestamp; full-path labels resolve at decision+12h",
                "selection": "no selection/rank/weight field is present; labels cover the full canonical population",
                "early_clean_nonflat": "side-relative close > +0.25 ATR and prefix MAE < 0.50 ATR",
            },
            "input_provenance": {
                "panel_manifest_sha256": sha256(PANEL_ROOT / "manifest.json"),
                "panel_sha256": panel_manifest["outputs_sha256"]["panel.parquet"],
                "path_manifest_sha256": sha256(PATH_ROOT / "manifest.json"),
                "paths_sha256": sha256(PATH_ROOT / "paths.parquet"),
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "All targets are future outcomes and are forbidden inference features.",
                "Best fixed horizon is a hindsight diagnostic class, not a directly deployable policy.",
                "Partial-profit and trailing actions require a separate path replay with explicit incremental cost accounting.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(safe(run(args.output)), indent=2))


if __name__ == "__main__":
    main()
