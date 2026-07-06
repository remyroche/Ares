#!/usr/bin/env python3
"""Materialize Stage176 current-replay short-exit labels.

Stage175 showed that the saved Stage167 first-touch columns are stale relative
to the current execution path store, while the Stage174 replay rows match a
fresh direct recomputation. This script turns a Stage174 replay policy into a
trainable label artifact with the standard label columns aligned to that
current-replay policy.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_aware import add_side_contract_columns  # noqa: E402
from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import _json_safe, _safe_mean, _safe_quantile  # noqa: E402


DEFAULT_SOURCE_LABELS_DIR = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_STAGE174_DIR = Path("data_perp/reports/stage174_short_exit_label_proxy_diagnostic_v1")
DEFAULT_STAGE175_SCORECARD = Path("data_perp/reports/stage175_first_touch_replay_alignment_scorecard_v1/summary.md")
DEFAULT_OUTPUT_RUN_ID = "20260703_151500_stage176_current_replay_fixed_hold6_labels"
DEFAULT_OUTPUT_DIR = Path("data_perp/artifacts") / DEFAULT_OUTPUT_RUN_ID / "labels"
DEFAULT_REPORT_DIR = Path("data_perp/reports/stage176_current_replay_fixed_hold6_labels_v1")
DEFAULT_SCORECARD_DIR = Path("data_perp/reports/stage176_current_replay_fixed_hold6_scorecard_v1")
DEFAULT_PRIMARY_POLICY = "contract_fixed_hold_6"
DEFAULT_AUX_POLICIES = (
    "contract_fixed_hold_4",
    "contract_fixed_hold_12",
    "contract_tp_sl_hold_24_tpmax_6",
    "contract_trail_static_act075_gb35_hold24",
    "contract_trail_decay_act075_min040_gb35_hold24",
    "label_first_touch_96",
)
CORE_LABEL_COLUMNS = (
    "__y_lbl__",
    "__y_bin__",
    "__y_ret__",
    "__y_outcome__",
    "__is_timeout__",
    "__tp__",
    "__sl__",
    "__u_policy_net__",
    "__r_policy_net__",
)
OUT_SL = np.int8(0)
OUT_TO = np.int8(1)
OUT_TP = np.int8(2)


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _safe_numeric(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.sum()) if len(series) else 0.0


def _sigmoid(values: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))


def _safe_name(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source_manifest(source_labels_dir: Path) -> dict[str, Any]:
    return _read_json(source_labels_dir / "labels_manifest.json")


def _label_files(source_labels_dir: Path) -> list[Path]:
    files = sorted(path for path in source_labels_dir.glob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found under {source_labels_dir}")
    return files


def _load_stage174_policy_rows(stage174_dir: Path, policies: list[str]) -> pd.DataFrame:
    path = stage174_dir / "stage174_policy_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = [
        "policy",
        "__ts__",
        "__symbol__",
        "barrier_pct",
        "tp_abs",
        "sl_abs",
        "finite_path",
        "net_return",
        "gross_return",
        "round_trip_cost",
        "exit_bars",
        "exit_hours",
        "exit_reason",
        "mfe_to_tp_until_exit",
        "mae_to_sl_until_exit",
        "max_favorable_return_until_exit",
        "max_adverse_return_until_exit",
        "peak_giveback_return",
        "peak_giveback_to_tp",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[frame["policy"].astype(str).isin(policies)].copy()
    if frame.empty:
        raise ValueError(f"No requested policies {policies} found in {path}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    if frame["__ts__"].isna().any():
        raise ValueError(f"{path} contains non-parseable __ts__ values")
    duplicates = int(frame.duplicated(["policy", "__ts__", "__symbol__"]).sum())
    if duplicates:
        raise ValueError(f"{path} has duplicate policy/key rows: {duplicates}")
    return frame.reset_index(drop=True)


def _policy_for_join(policy_rows: pd.DataFrame, policy: str, prefix: str) -> pd.DataFrame:
    selected = policy_rows[policy_rows["policy"].astype(str).eq(policy)].copy()
    if selected.empty:
        raise ValueError(f"Missing policy rows for {policy}")
    rename = {
        "policy": f"{prefix}_policy",
        "barrier_pct": f"{prefix}_barrier_pct",
        "tp_abs": f"{prefix}_tp_abs",
        "sl_abs": f"{prefix}_sl_abs",
        "finite_path": f"{prefix}_finite_path",
        "net_return": f"{prefix}_net",
        "gross_return": f"{prefix}_gross",
        "round_trip_cost": f"{prefix}_round_trip_cost",
        "exit_bars": f"{prefix}_exit_bars",
        "exit_hours": f"{prefix}_exit_hours",
        "exit_reason": f"{prefix}_exit_reason",
        "mfe_to_tp_until_exit": f"{prefix}_mfe_to_tp_until_exit",
        "mae_to_sl_until_exit": f"{prefix}_mae_to_sl_until_exit",
        "max_favorable_return_until_exit": f"{prefix}_max_favorable_return_until_exit",
        "max_adverse_return_until_exit": f"{prefix}_max_adverse_return_until_exit",
        "peak_giveback_return": f"{prefix}_peak_giveback_return",
        "peak_giveback_to_tp": f"{prefix}_peak_giveback_to_tp",
    }
    return selected.rename(columns=rename)


def _monthly_summary(frame: pd.DataFrame, *, net_col: str, reason_col: str, mae_col: str, exit_col: str) -> list[dict[str, Any]]:
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    months = ts.dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for period, idx in pd.Series(np.arange(len(frame)), index=frame.index).groupby(months, dropna=False):
        group = frame.iloc[idx.to_numpy(dtype=np.int64)]
        net = _safe_numeric(group[net_col])
        reason = group[reason_col].astype(str)
        rows.append(
            {
                "period": str(period),
                "rows": int(len(group)),
                "sum_net": _safe_sum(net),
                "mean_net": _safe_mean(net),
                "q10_net": _safe_quantile(net, 0.10),
                "win_rate": _safe_mean(net > 0.0),
                "y_pos_rate": _safe_mean(_safe_numeric(group["__y_bin__"]) > 0.5),
                "exit_bars_p90": _safe_quantile(group[exit_col], 0.90),
                "mae_to_sl_p90": _safe_quantile(group[mae_col], 0.90),
                "fixed_exit_rate": _safe_mean(reason.str.startswith("fixed_hold")),
                "ineligible_barrier_rate": _safe_mean(reason.eq("ineligible_barrier")),
                "missing_path_rate": _safe_mean(reason.eq("missing_path")),
            }
        )
    return rows


def _artifact_summary(frame: pd.DataFrame, *, policy_prefix: str) -> dict[str, Any]:
    net = _safe_numeric(frame[f"{policy_prefix}_net"])
    reason = frame[f"{policy_prefix}_exit_reason"].astype(str)
    return {
        "rows": int(len(frame)),
        "timestamp_min": pd.to_datetime(frame["__ts__"], errors="coerce").min(),
        "timestamp_max": pd.to_datetime(frame["__ts__"], errors="coerce").max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "finite_net_frac": float(np.isfinite(net.to_numpy(dtype=np.float64)).mean()) if len(net) else 0.0,
        "mean_net": _safe_mean(net),
        "sum_net": _safe_sum(net),
        "q10_net": _safe_quantile(net, 0.10),
        "q90_net": _safe_quantile(net, 0.90),
        "win_rate": _safe_mean(net > 0.0),
        "target_soft_mean": _safe_mean(frame["__stage176_target_soft__"]),
        "target_hard_rate": _safe_mean(frame["__stage176_target_hard__"]),
        "econ_target_soft_mean": _safe_mean(frame["__stage176_econ_target_soft__"]),
        "econ_target_hard_rate": _safe_mean(frame["__stage176_econ_target_hard__"]),
        "exit_bars_p90": _safe_quantile(frame[f"{policy_prefix}_exit_bars"], 0.90),
        "mae_to_sl_p90": _safe_quantile(frame[f"{policy_prefix}_mae_to_sl_until_exit"], 0.90),
        "fixed_exit_rate": _safe_mean(reason.str.startswith("fixed_hold")),
        "ineligible_barrier_rate": _safe_mean(reason.eq("ineligible_barrier")),
        "monthly": _monthly_summary(
            frame,
            net_col=f"{policy_prefix}_net",
            reason_col=f"{policy_prefix}_exit_reason",
            mae_col=f"{policy_prefix}_mae_to_sl_until_exit",
            exit_col=f"{policy_prefix}_exit_bars",
        ),
    }


def _copy_source_core_columns(out: pd.DataFrame) -> None:
    for col in CORE_LABEL_COLUMNS:
        if col in out.columns:
            out[f"__stage176_source{col}"] = out[col].to_numpy(copy=False)


def _materialize_frame(
    source: pd.DataFrame,
    primary: pd.DataFrame,
    aux: dict[str, pd.DataFrame],
    *,
    primary_policy: str,
    target_center: float,
    target_temperature: float,
    econ_temperature: float,
) -> pd.DataFrame:
    source = source.copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], errors="coerce")
    source["__symbol__"] = source["__symbol__"].astype(str)
    joined = source.merge(primary, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    missing = int(joined["__stage176_primary_net"].isna().sum())
    if missing:
        raise ValueError(f"Missing primary policy rows after join: {missing}")
    for policy, frame in aux.items():
        prefix = f"__stage176_aux_{_safe_name(policy)}"
        aux_join = frame[["__ts__", "__symbol__", f"{prefix}_net", f"{prefix}_exit_bars", f"{prefix}_mae_to_sl_until_exit"]].copy()
        joined = joined.merge(aux_join, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")

    out = joined.copy()
    _copy_source_core_columns(out)
    net = _safe_numeric(out["__stage176_primary_net"]).fillna(0.0).to_numpy(dtype=np.float64)
    hard = (net > 0.0).astype(np.float32)
    label_code = np.where(hard > 0.5, OUT_TP, OUT_SL).astype(np.int8)
    finite = _safe_numeric(out["__stage176_primary_finite_path"]).fillna(0.0).to_numpy(dtype=np.float32)
    reason = out["__stage176_primary_exit_reason"].astype(str)
    ineligible = reason.eq("ineligible_barrier").to_numpy()
    missing_path = reason.eq("missing_path").to_numpy()

    target_soft = _sigmoid((np.clip(net, -0.03, 0.06) - float(target_center)) / max(float(target_temperature), 1e-12))
    mae = _safe_numeric(out["__stage176_primary_mae_to_sl_until_exit"]).fillna(10.0)
    exit_bars = _safe_numeric(out["__stage176_primary_exit_bars"]).fillna(96.0)
    barrier = _safe_numeric(out["__stage176_primary_barrier_pct"]).fillna(0.10)
    econ_utility = (
        pd.Series(np.clip(net, -0.03, 0.06), index=out.index)
        - 0.0040 * (mae - 1.0).clip(lower=0.0)
        - 0.00020 * (exit_bars - 8.0).clip(lower=0.0)
        - 0.35 * (barrier - 0.030).clip(lower=0.0)
    )
    econ_soft = _sigmoid(econ_utility / max(float(econ_temperature), 1e-12))
    econ_hard = (_safe_numeric(econ_utility) > 0.0).astype(np.float32)

    out["__y_lbl__"] = label_code
    out["__y_outcome__"] = label_code.astype(np.int8)
    out["__y_bin__"] = hard.astype(np.float32)
    out["__y_ret__"] = net.astype(np.float32)
    out["__is_timeout__"] = np.zeros(len(out), dtype=np.float32)
    out["__tp__"] = _safe_numeric(out["__stage176_primary_tp_abs"]).to_numpy(dtype=np.float32)
    out["__sl__"] = _safe_numeric(out["__stage176_primary_sl_abs"]).to_numpy(dtype=np.float32)
    out["__u_policy_net__"] = net.astype(np.float32)
    out["__r_policy_net__"] = net.astype(np.float32)
    if "__w__" not in out.columns:
        out["__w__"] = np.ones(len(out), dtype=np.float32)

    out["__stage176_policy_name__"] = str(primary_policy)
    out["__stage176_target_soft__"] = target_soft.astype(np.float32)
    out["__stage176_target_hard__"] = hard.astype(np.float32)
    out["__stage176_econ_utility__"] = _safe_numeric(econ_utility).to_numpy(dtype=np.float32)
    out["__stage176_econ_target_soft__"] = econ_soft.astype(np.float32)
    out["__stage176_econ_target_hard__"] = econ_hard.to_numpy(dtype=np.float32)
    out["__stage176_finite_path__"] = finite.astype(np.float32)
    out["__stage176_ineligible_barrier__"] = ineligible.astype(np.float32)
    out["__stage176_missing_path__"] = missing_path.astype(np.float32)

    out = add_side_contract_columns(
        out,
        side="long",
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe="1h",
        copy=False,
    )
    return out


def _write_report(
    *,
    report_dir: Path,
    scorecard_dir: Path,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    monthly_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for row in summary.get("monthly", []):
            out = {"dataset": summary["dataset"]}
            out.update(row)
            monthly_rows.append(out)
    monthly = pd.DataFrame(monthly_rows)
    monthly_path = report_dir / "stage176_monthly_summary.csv"
    monthly.to_csv(monthly_path, index=False)
    manifest["outputs"]["monthly_summary"] = str(monthly_path)

    summary_cols = [
        "dataset",
        "rows",
        "mean_net",
        "win_rate",
        "target_soft_mean",
        "econ_target_soft_mean",
        "exit_bars_p90",
        "mae_to_sl_p90",
        "ineligible_barrier_rate",
    ]
    monthly_cols = [
        "dataset",
        "period",
        "rows",
        "mean_net",
        "sum_net",
        "win_rate",
        "exit_bars_p90",
        "mae_to_sl_p90",
        "ineligible_barrier_rate",
    ]
    summary_frame = pd.DataFrame(summaries)
    lines = [
        "# Stage176 Current-Replay Short-Exit Label Artifact",
        "",
        "Scope: materialized trainable label candidate. The standard label columns now point to the current-replay fixed-hold policy, while previous core labels are preserved under `__stage176_source...` columns.",
        "",
        f"Output labels: `{manifest['output_labels_dir']}`",
        f"Primary policy: `{manifest['primary_policy']}`",
        f"Source labels: `{manifest['source_labels_dir']}`",
        f"Stage174 replay source: `{manifest['stage174_policy_rows']}`",
        f"Stage175 alignment evidence: `{manifest['stage175_scorecard']}`",
        "",
        "## Dataset Summary",
        "",
        _table(summary_frame, summary_cols, limit=50),
        "",
        "## Monthly Summary",
        "",
        _table(monthly, monthly_cols, limit=80),
        "",
        "## Outputs",
        "",
    ]
    report_path = report_dir / "stage176_current_replay_fixed_hold6_labels.md"
    scorecard_path = scorecard_dir / "summary.md"
    manifest["outputs"]["markdown"] = str(report_path)
    manifest["outputs"]["scorecard"] = str(scorecard_path)
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    score_lines = [
        "# Stage176 Scorecard - Current-Replay Fixed-Hold-6 Labels",
        "",
        "## Finding",
        "",
        "A trainable label artifact has been materialized from the current Stage174 replay rows using `contract_fixed_hold_6` as the primary target. This avoids the stale saved Stage167 first-touch PnL identified in Stage175.",
        "",
        "## Label Contract",
        "",
        "- `__y_ret__`, `__u_policy_net__`, and `__r_policy_net__` equal `contract_fixed_hold_6` net return after the configured round-trip cost.",
        "- `__y_bin__` and `__y_outcome__` are binary-positive-net targets: positive net maps to outcome `2`; non-positive net maps to outcome `0`.",
        "- Original core label columns are preserved as `__stage176_source__...` columns.",
        "",
        "## Dataset Summary",
        "",
        _table(summary_frame, summary_cols, limit=50),
        "",
        "## Monthly Summary",
        "",
        _table(monthly, monthly_cols, limit=80),
        "",
        "## Next Gate",
        "",
        "Run a no-training label readiness check and then a small base/meta smoke test before promoting this as a default training label. Do not compare its PnL to old saved first-touch columns except as stale-artifact context.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        score_lines.append(f"- {key}: `{value}`")
    scorecard_path.write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    return report_path, scorecard_path


def run_materialization(
    *,
    source_labels_dir: Path,
    stage174_dir: Path,
    stage175_scorecard: Path,
    output_run_id: str,
    output_labels_dir: Path,
    report_dir: Path,
    scorecard_dir: Path,
    primary_policy: str,
    auxiliary_policies: list[str],
    target_center: float,
    target_temperature: float,
    econ_temperature: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_labels_dir.exists() and any(output_labels_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_labels_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _source_manifest(source_labels_dir)
    policies = list(dict.fromkeys([primary_policy, *auxiliary_policies]))
    stage174_rows = _load_stage174_policy_rows(stage174_dir, policies)
    primary = _policy_for_join(stage174_rows, primary_policy, "__stage176_primary")
    aux = {
        policy: _policy_for_join(stage174_rows, policy, f"__stage176_aux_{_safe_name(policy)}")
        for policy in auxiliary_policies
        if policy != primary_policy
    }

    dataset_meta: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for source_file in _label_files(source_labels_dir):
        source = pd.read_parquet(source_file).reset_index(drop=True)
        if "__ts__" not in source.columns or "__symbol__" not in source.columns:
            raise ValueError(f"{source_file} is missing __ts__/__symbol__")
        out = _materialize_frame(
            source,
            primary,
            aux,
            primary_policy=primary_policy,
            target_center=target_center,
            target_temperature=target_temperature,
            econ_temperature=econ_temperature,
        )
        output_file = output_labels_dir / source_file.name
        out.to_parquet(output_file, index=False)
        summary = _artifact_summary(out, policy_prefix="__stage176_primary")
        summary.update(
            {
                "dataset": source_file.stem,
                "source_file": str(source_file),
                "output_file": str(output_file),
                "primary_policy": str(primary_policy),
            }
        )
        summaries.append(summary)
        dataset_meta[source_file.stem] = {
            "file": source_file.name,
            "rows": int(summary["rows"]),
            "timestamp_min": summary["timestamp_min"],
            "timestamp_max": summary["timestamp_max"],
            "symbols": int(summary["symbols"]),
            "columns": list(out.columns),
        }

    manifest = {
        **{key: value for key, value in source_manifest.items() if key != "datasets"},
        "run_id": str(output_run_id),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": str(source_labels_dir / "labels_manifest.json"),
        "output_labels_dir": str(output_labels_dir),
        "primary_policy": str(primary_policy),
        "auxiliary_policies": list(auxiliary_policies),
        "stage174_dir": str(stage174_dir),
        "stage174_policy_rows": str(stage174_dir / "stage174_policy_rows.csv"),
        "stage174_manifest": str(stage174_dir / "manifest.json"),
        "stage175_scorecard": str(stage175_scorecard),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_center": float(target_center),
        "target_temperature": float(target_temperature),
        "econ_temperature": float(econ_temperature),
        "label_contract": {
            "__y_ret__": f"{primary_policy}.net_return",
            "__u_policy_net__": f"{primary_policy}.net_return",
            "__r_policy_net__": f"{primary_policy}.net_return",
            "__y_bin__": "net_return > 0",
            "__y_outcome__": "2 if net_return > 0 else 0",
            "__is_timeout__": "0; fixed-hold time exit is represented in __stage176_primary_exit_reason__",
        },
        "datasets": dataset_meta,
        "outputs": {
            "labels_manifest": str(output_labels_dir / "labels_manifest.json"),
            "materialization_summary": str(output_labels_dir / "stage176_current_replay_label_materialization_summary.json"),
        },
    }
    report_path, scorecard_path = _write_report(
        report_dir=report_dir,
        scorecard_dir=scorecard_dir,
        manifest=manifest,
        summaries=summaries,
    )
    manifest["outputs"]["markdown"] = str(report_path)
    manifest["outputs"]["scorecard"] = str(scorecard_path)
    (output_labels_dir / "labels_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    (output_labels_dir / "stage176_current_replay_label_materialization_summary.json").write_text(
        json.dumps(_json_safe({"datasets": summaries}), indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels-dir", type=Path, default=DEFAULT_SOURCE_LABELS_DIR)
    parser.add_argument("--stage174-dir", type=Path, default=DEFAULT_STAGE174_DIR)
    parser.add_argument("--stage175-scorecard", type=Path, default=DEFAULT_STAGE175_SCORECARD)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--output-labels-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--primary-policy", default=DEFAULT_PRIMARY_POLICY)
    parser.add_argument("--auxiliary-policies", default=",".join(DEFAULT_AUX_POLICIES))
    parser.add_argument("--target-center", type=float, default=0.0)
    parser.add_argument("--target-temperature", type=float, default=0.010)
    parser.add_argument("--econ-temperature", type=float, default=0.010)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_materialization(
        source_labels_dir=args.source_labels_dir,
        stage174_dir=args.stage174_dir,
        stage175_scorecard=args.stage175_scorecard,
        output_run_id=str(args.output_run_id),
        output_labels_dir=args.output_labels_dir,
        report_dir=args.report_dir,
        scorecard_dir=args.scorecard_dir,
        primary_policy=str(args.primary_policy),
        auxiliary_policies=_parse_csv(args.auxiliary_policies, DEFAULT_AUX_POLICIES),
        target_center=float(args.target_center),
        target_temperature=float(args.target_temperature),
        econ_temperature=float(args.econ_temperature),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
