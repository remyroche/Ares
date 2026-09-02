#!/usr/bin/env python3
"""Materialize exact deployed-exit capture labels from signed 1m paths.

The favorable path is truncated at the deployed policy exit minute.  This
prevents post-exit MFE from masquerading as an opportunity the policy could
have captured.  Labels remain conservatively available only at the canonical
12-hour resolution timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_febapr2025_exact1m_path_head_labels import (
    _decode_paths,
    _execution_adjusted_path,
)


RELIABILITY = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v2"
PATH_ROOT = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1"
PATH_LABEL_ROOT = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_path_head_labels_20260727_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_pre_exit_capture_labels_20260730_v2"
IDENTITY = ("candidate_id", "side_name")
EXPECTED_ROWS = 110_730
HORIZON_MINUTES = 720
MIN_MEANINGFUL_MFE_ATR = 1.5
MIN_MEANINGFUL_MFE_RETURN = 0.015
EPSILON = 1e-8


class CaptureLabelError(RuntimeError):
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
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(safe(dict(payload)), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_reliability(root: Path) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file() or not seal.is_file():
        raise CaptureLabelError("sealed reliability input is missing")
    if sha256(manifest) != seal.read_text().split()[0]:
        raise CaptureLabelError("reliability manifest seal mismatch")
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != "canonical_execution_reliability_input_v2":
        raise CaptureLabelError("wrong reliability input")
    if sha256(root / "panel.parquet") != payload["outputs_sha256"]["panel.parquet"]:
        raise CaptureLabelError("reliability panel hash mismatch")
    return payload


def verify_paths(path_root: Path, signed_labels_root: Path) -> dict[str, Any]:
    manifest_path = path_root / "manifest.json"
    paths_path = path_root / "paths.parquet"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "execution_entry_timing_1m_paths_v1":
        raise CaptureLabelError("wrong exact-path source")
    if manifest.get("rows", {}).get("output") != 205_194:
        raise CaptureLabelError("exact-path source row count drift")
    if manifest.get("timing", {}).get("path_minutes") != HORIZON_MINUTES:
        raise CaptureLabelError("exact path is not 720 minutes")
    shard_manifest = signed_labels_root / "shards/part-000/manifest.json"
    signed = json.loads(shard_manifest.read_text())
    declared = signed["sources"]["exact_1m_paths"]
    if Path(str(declared["path"])).resolve() != paths_path.resolve():
        raise CaptureLabelError("signed path source differs")
    actual = sha256(paths_path)
    if actual != str(declared["sha256"]):
        raise CaptureLabelError("exact-path source hash mismatch")
    if str(manifest.get("source_artifact_sha256")) != actual:
        raise CaptureLabelError("path manifest source-artifact hash mismatch")
    return {
        "path_manifest": sha256(manifest_path),
        "paths": actual,
        "signed_path_label_manifest": sha256(shard_manifest),
    }


def canonicalize_path_symbol(values: pd.Series) -> pd.Series:
    """Map signed path slash notation to the canonical underscore identity."""
    return values.astype(str).str.replace("/", "_", regex=False)


def capture_columns(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side_name: np.ndarray,
    entry_spread_bps: np.ndarray,
    exit_spread_bps: np.ndarray,
    exit_minute: np.ndarray,
    atr_fraction: np.ndarray,
    gross: np.ndarray,
    cost: np.ndarray,
    net: np.ndarray,
) -> dict[str, np.ndarray]:
    sign = np.where(np.asarray(side_name, dtype=str) == "short", -1.0, 1.0)
    if not np.isin(np.asarray(side_name, dtype=str), ("long", "short")).all():
        raise CaptureLabelError("noncanonical side")
    entry, executable_high, executable_low, _ = _execution_adjusted_path(
        open_,
        high,
        low,
        close,
        side_sign=sign,
        entry_spread_bps=np.asarray(entry_spread_bps, dtype=float),
        exit_spread_bps=np.asarray(exit_spread_bps, dtype=float),
    )
    minute = np.rint(np.asarray(exit_minute, dtype=float)).astype(int)
    if np.any(minute < 0) or np.any(minute >= HORIZON_MINUTES):
        raise CaptureLabelError("deployed exit minute escapes exact path")
    pre_exit_mfe = np.empty(len(entry), dtype=float)
    for row, last in enumerate(minute):
        if sign[row] > 0:
            favorable = executable_high[row, : last + 1] / entry[row] - 1.0
        else:
            favorable = 1.0 - executable_low[row, : last + 1] / entry[row]
        pre_exit_mfe[row] = max(0.0, float(np.max(favorable)))
    gross = np.asarray(gross, dtype=float)
    cost = np.asarray(cost, dtype=float)
    net = np.asarray(net, dtype=float)
    if np.max(np.abs(gross - cost - net)) > 1e-7:
        raise CaptureLabelError("gross-cost-net accounting failed")
    # Some trailing exits come from a policy-price lineage that is slightly
    # above the independently signed 1m executable path.  Do not repair or
    # clip those rows into apparent capture.  Preserve their path opportunity
    # labels, fail the capture-parity mask, and exclude them from capture-head
    # training.
    parity = gross <= pre_exit_mfe + 1e-5
    meaningful_floor = np.maximum(
        MIN_MEANINGFUL_MFE_ATR * np.asarray(atr_fraction, dtype=float),
        MIN_MEANINGFUL_MFE_RETURN,
    )
    meaningful = pre_exit_mfe >= meaningful_floor
    economic = pre_exit_mfe > cost
    capture_valid = economic & parity
    positive_gross = np.maximum(gross, 0.0)
    positive_net = np.maximum(net, 0.0)
    capture_ratio = np.divide(
        positive_gross,
        np.maximum(pre_exit_mfe, EPSILON),
    )
    net_opportunity = np.maximum(pre_exit_mfe - cost, 0.0)
    economic_capture_ratio = np.divide(
        positive_net,
        np.maximum(net_opportunity, EPSILON),
    )
    capture_ratio = np.where(parity, np.clip(capture_ratio, 0.0, 1.0), np.nan)
    economic_capture_ratio = np.where(
        capture_valid,
        np.clip(economic_capture_ratio, 0.0, 1.0),
        np.nan,
    )
    return {
        "pre_exit_mfe_return": pre_exit_mfe.astype(np.float32),
        "pre_exit_mfe_atr": np.divide(
            pre_exit_mfe,
            np.maximum(np.asarray(atr_fraction, dtype=float), EPSILON),
        ).astype(np.float32),
        "target_pre_exit_meaningful_mfe": meaningful.astype(np.int8),
        "target_pre_exit_economic_opportunity": economic.astype(np.int8),
        "pre_exit_path_policy_parity": parity.astype(np.int8),
        "target_pre_exit_capture_valid": capture_valid.astype(np.int8),
        "target_pre_exit_capture_net_positive": (net > 0).astype(np.int8),
        "target_pre_exit_capture_ratio": capture_ratio.astype(np.float32),
        "target_pre_exit_economic_capture_ratio": economic_capture_ratio.astype(np.float32),
        "target_pre_exit_capture_shortfall_return": np.where(
            parity, np.maximum(pre_exit_mfe - positive_gross, 0.0), np.nan
        ).astype(np.float32),
        "target_pre_exit_uncaptured_net_opportunity_return": np.where(
            capture_valid,
            np.maximum(net_opportunity - positive_net, 0.0),
            np.nan,
        ).astype(np.float32),
    }


def materialize_batch(paths: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    joined = paths.merge(context, on=list(IDENTITY), how="inner", validate="one_to_one")
    if joined.empty:
        return joined
    if not pd.to_datetime(joined.__ts___x, utc=True).eq(
        pd.to_datetime(joined.__ts___y, utc=True)
    ).all():
        raise CaptureLabelError("path/context signal timestamp mismatch")
    normalized_symbol = canonicalize_path_symbol(joined.__symbol___x)
    if not normalized_symbol.eq(joined.__symbol___y.astype(str)).all():
        raise CaptureLabelError("normalized path/context symbol mismatch")
    open_, high, low, close = _decode_paths(joined.execution_future_path)
    encoded_start = np.asarray(
        [json.loads(value)["timestamp"][0] for value in joined.execution_future_path],
        dtype=np.int64,
    )
    decision = pd.to_datetime(joined.execution_decision_utc, utc=True)
    if not np.array_equal(encoded_start, decision.astype("int64").to_numpy()):
        raise CaptureLabelError("exact path does not start at decision")
    atr_fraction = joined.atr_1h.to_numpy(float) / joined.decision_price.to_numpy(float)
    targets = capture_columns(
        open_=open_,
        high=high,
        low=low,
        close=close,
        side_name=joined.side_name.to_numpy(),
        entry_spread_bps=joined.entry_spread.to_numpy(float),
        exit_spread_bps=joined.exit_spread.to_numpy(float),
        exit_minute=joined.execution_exit_minute.to_numpy(float),
        atr_fraction=atr_fraction,
        gross=joined.execution_gross_ev_12h.to_numpy(float),
        cost=joined.execution_cost_return.to_numpy(float),
        net=joined.execution_net_ev_12h.to_numpy(float),
    )
    output = joined.loc[
        :,
        [
            "candidate_id",
            "side_name",
            "__symbol___y",
            "__ts___y",
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_exit_minute",
            "execution_exit_reason",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
        ],
    ].rename(columns={"__symbol___y": "__symbol__", "__ts___y": "__ts__"})
    output["label_available_at_utc"] = output.execution_label_end_utc
    output["path_atr_fraction"] = atr_fraction.astype(np.float32)
    for name, values in targets.items():
        output[name] = values
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    reliability_manifest = verify_reliability(args.reliability)
    path_hashes = verify_paths(args.path_root, args.path_label_root)
    context_columns = [
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_exit_minute",
        "execution_exit_reason",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    ]
    context = pd.read_parquet(
        args.reliability / "panel.parquet", columns=context_columns
    )
    if len(context) != EXPECTED_ROWS or context.duplicated(list(IDENTITY)).any():
        raise CaptureLabelError("reliability identity contract failed")
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        context[column] = pd.to_datetime(context[column], utc=True, errors="raise")
    identifiers = set(context.candidate_id.astype(str))
    pieces: list[pd.DataFrame] = []
    source = pq.ParquetFile(args.path_root / "paths.parquet")
    columns = [
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        "execution_future_path",
        "atr_1h",
        "decision_price",
        "entry_spread",
        "exit_spread",
    ]
    for batch in source.iter_batches(batch_size=int(args.batch_rows), columns=columns):
        raw = batch.to_pandas()
        selected = raw.loc[raw.candidate_id.astype(str).isin(identifiers)]
        if selected.empty:
            continue
        selected_context = context.loc[
            context.candidate_id.astype(str).isin(set(selected.candidate_id.astype(str)))
        ]
        pieces.append(materialize_batch(selected, selected_context))
    labels = pd.concat(pieces, ignore_index=True)
    if len(labels) != EXPECTED_ROWS or labels.duplicated(list(IDENTITY)).any():
        raise CaptureLabelError(
            f"capture coverage failed: {len(labels)} rows"
        )
    if set(labels.candidate_id.astype(str)) != identifiers:
        raise CaptureLabelError("capture identity set differs from reliability input")
    labels = labels.sort_values(
        ["__ts__", "__symbol__", "side_name", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        labels.to_parquet(stage / "labels.parquet", index=False, compression="zstd")
        support = (
            labels.assign(month=labels.__ts__.dt.strftime("%Y-%m"))
            .groupby(["month", "side_name"], sort=True)
            .agg(
                rows=("candidate_id", "size"),
                pre_exit_meaningful_mfe=("target_pre_exit_meaningful_mfe", "sum"),
                pre_exit_economic_opportunity=("target_pre_exit_economic_opportunity", "sum"),
                path_policy_parity_rows=("pre_exit_path_policy_parity", "sum"),
                capture_valid_rows=("target_pre_exit_capture_valid", "sum"),
                captured_net_positive=("target_pre_exit_capture_net_positive", "sum"),
                mean_capture_ratio=("target_pre_exit_capture_ratio", "mean"),
                mean_economic_capture_ratio=("target_pre_exit_economic_capture_ratio", "mean"),
            )
            .reset_index()
        )
        support.to_csv(stage / "support_by_month_side.csv", index=False)
        outputs = {
            path.name: sha256(path) for path in stage.iterdir() if path.is_file()
        }
        manifest = {
            "schema": "canonical_pre_exit_capture_labels_v2",
            "run_id": args.output_dir.name,
            "status": "SEALED_TARGET_ONLY_NO_MODEL_SELECTION_NO_PROMOTION",
            "rows": len(labels),
            "identity": list(IDENTITY),
            "timing": {
                "path_start": "execution decision",
                "path_stop": "deployed execution_exit_minute inclusive",
                "label_available_at": "canonical decision + 12h resolution timestamp",
            },
            "path_contract": {
                "cadence_minutes": 1,
                "maximum_minutes": HORIZON_MINUTES,
                "entry_spread": "applied once to entry",
                "exit_spread": "applied once to every candidate exit price",
                "post_exit_prices": "excluded",
                "symbol_identity": "signed path '/' notation normalized exactly to canonical '_' notation and asserted rowwise",
            },
            "target_contract": {
                "meaningful_mfe": "pre-exit MFE >= max(1.5*ATR fraction, 1.5%)",
                "economic_opportunity": "pre-exit executable MFE > canonical row cost",
                "capture_valid": "economic opportunity and exact path-policy parity rows only",
                "path_policy_parity": "gross <= exact pre-exit executable MFE + 1e-5; failures are excluded, never clipped or imputed",
                "capture_net_positive": "canonical deployed-policy net > 0; interpret only where capture_valid=1",
                "capture_ratio": "clip(max(gross,0)/pre-exit executable MFE,0,1)",
                "economic_capture_ratio": "clip(max(net,0)/max(pre-exit MFE-cost,epsilon),0,1); interpret only where capture_valid=1",
                "shortfall": "pre-exit executable opportunity not retained by deployed exit",
            },
            "source_sha256": {
                "reliability_manifest": sha256(args.reliability / "manifest.json"),
                "reliability_panel": reliability_manifest["outputs_sha256"]["panel.parquet"],
                **path_hashes,
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--reliability", type=Path, default=RELIABILITY)
    command.add_argument("--path-root", type=Path, default=PATH_ROOT)
    command.add_argument("--path-label-root", type=Path, default=PATH_LABEL_ROOT)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--batch-rows", type=int, default=500)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
