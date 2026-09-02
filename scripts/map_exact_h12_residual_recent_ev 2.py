#!/usr/bin/env python3
"""Apply the causal 21-day EV admission map to frozen exact-H12 score arms.

No predictive model is refit.  Each daily map uses only exact-H12 labels
resolved before that UTC-day snapshot.  Final evaluation is one pooled-global
monthly top-k after the mapped score, with the frozen raw score used only to
resolve isotonic plateaus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings
from scripts.run_exact_h12_side_local_residual_oof import stable_top


SCHEMA = "exact_h12_residual_recent_ev_mapping_v1"
INPUT_SCHEMA = "exact_h12_side_local_residual_oof_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
NET = "execution_net_ev_12h"
WINDOW_DAYS = 21
MIN_REFERENCE_ROWS = 500
SIDE_SUPPORT_TARGET = 500.0
DEFAULT_RESIDUAL = (
    ROOT
    / "data_perp/artifacts/exact_h12_side_local_residual_oof_20260730_v2"
)
DEFAULT_WATERFALL = (
    ROOT
    / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/exact_h12_residual_recent_ev_mapping_20260730_v1"
)
SCORES: Mapping[str, str] = {
    "base_alpha_raw": "score_base_alpha",
    "base_ev_exact_h12": "score_exact_h12_base_ev_bps",
    "residual_exact_h12": "score_exact_h12_residual_bps",
    "direct_q25_exact_h12": "score_direct_q25_challenger_bps",
    "residual_legacy_24h": "score_residual_expected_ev",
}


class RecentMappingError(RuntimeError):
    """Raised when frozen-score causal mapping cannot be proven."""


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
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verify_input(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.sha256"
    if sha256(manifest_path) != digest_path.read_text().split()[0]:
        raise RecentMappingError("exact-H12 residual manifest seal changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != INPUT_SCHEMA
        or manifest.get("promotion_eligible") is not False
    ):
        raise RecentMappingError("unexpected exact-H12 residual input")
    for record in manifest.get("outputs", {}).values():
        path = Path(str(record["path"]))
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RecentMappingError(f"exact-H12 residual output changed: {path}")
    return manifest


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = (
        result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    )
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    return result


def _load(
    residual_root: Path,
    waterfall_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    residual_manifest = _verify_input(residual_root)
    oof_path = residual_root / "oof_predictions.parquet"
    waterfall_path = waterfall_root / "allscore_waterfall.parquet"
    waterfall_manifest_path = waterfall_root / "manifest.json"
    waterfall_manifest = json.loads(
        waterfall_manifest_path.read_text(encoding="utf-8")
    )
    if (
        waterfall_manifest.get("outputs", {})
        .get("allscore_waterfall", {})
        .get("sha256")
        != sha256(waterfall_path)
    ):
        raise RecentMappingError("waterfall binding changed")
    oof = _normalise(pd.read_parquet(oof_path))
    waterfall = _normalise(pd.read_parquet(waterfall_path))
    frame = waterfall.merge(
        oof.loc[
            :,
            [
                *IDENTITY,
                "score_exact_h12_base_ev_bps",
                "score_exact_h12_residual_bps",
                "residual_oof_fold",
                "is_strict_oof",
            ],
        ],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if (
        len(frame) != 127_777
        or frame["candidate_id"].duplicated().any()
        or not frame["is_strict_oof"].all()
    ):
        raise RecentMappingError("frozen score identity/OOF coverage changed")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not frame["execution_decision_utc"].eq(
        frame["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise RecentMappingError("decision timing changed")
    if not frame["execution_label_end_utc"].eq(
        frame["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise RecentMappingError("exact-H12 endpoint changed")
    if not np.allclose(
        frame["execution_gross_ev_12h"] - frame["execution_cost_return"],
        frame[NET],
        rtol=0.0,
        atol=1e-10,
    ):
        raise RecentMappingError("gross-cost-net reconciliation failed")
    missing = sorted(set(SCORES.values()).difference(frame.columns))
    if missing or not np.isfinite(frame[list(SCORES.values())].to_numpy(float)).all():
        raise RecentMappingError(f"frozen score columns invalid: {missing}")
    evidence = {
        "residual_manifest": binding(residual_root / "manifest.json"),
        "residual_manifest_schema": residual_manifest["schema"],
        "oof_predictions": binding(oof_path),
        "waterfall": binding(waterfall_path),
        "waterfall_manifest": binding(waterfall_manifest_path),
        "rows": len(frame),
    }
    return frame, evidence


def _map_scores(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    audits: list[dict[str, Any]] = []
    for score_name, score_column in SCORES.items():
        mapped, audit = causal_mappings(
            output,
            score_col=score_column,
            window_days=WINDOW_DAYS,
            min_reference_rows=MIN_REFERENCE_ROWS,
            side_support_target=SIDE_SUPPORT_TARGET,
        )
        mapped_column = f"mapped_{score_name}_ev"
        output[mapped_column] = mapped["causal_recent_side_isotonic_ev"]
        for record in audit:
            audits.append({"score_name": score_name, **record})
    mapped_columns = [f"mapped_{name}_ev" for name in SCORES]
    output["common_mapping_eligible"] = output[mapped_columns].notna().all(axis=1)
    if output.loc[output["common_mapping_eligible"], mapped_columns].isna().any().any():
        raise RecentMappingError("common mapped surface contains missing values")
    return output, pd.DataFrame(audits)


def _evaluate(
    mapped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = mapped.loc[mapped["common_mapping_eligible"]].copy()
    work["candidate_month"] = work["__ts__"].dt.strftime("%Y-%m")
    metrics: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    ties: list[dict[str, Any]] = []
    for month, month_rows in work.groupby("candidate_month", sort=True):
        for score_name, raw_column in SCORES.items():
            mapped_column = f"mapped_{score_name}_ev"
            for mode, score_column, secondary in (
                ("raw_common", raw_column, None),
                ("recent_ev_mapped", mapped_column, raw_column),
            ):
                record: dict[str, Any] = {
                    "month": str(month),
                    "score_name": score_name,
                    "mode": mode,
                    "score_column": score_column,
                    "eligible_rows": len(month_rows),
                }
                for fraction in (0.01, 0.05, 0.10, 0.20):
                    selected = stable_top(
                        month_rows,
                        score_column,
                        fraction,
                        secondary_column=secondary,
                    )
                    label = f"top{int(fraction * 100):02d}"
                    net = selected[NET].to_numpy(float) * 1e4
                    record.update(
                        {
                            f"{label}_rows": len(selected),
                            f"{label}_gross_bps": float(
                                selected["execution_gross_ev_12h"].mean() * 1e4
                            ),
                            f"{label}_cost_bps": float(
                                selected["execution_cost_return"].mean() * 1e4
                            ),
                            f"{label}_net_bps": float(net.mean()),
                            f"{label}_positive_net_rate": float((net > 0).mean()),
                        }
                    )
                    if fraction == 0.10:
                        book = selected.loc[
                            :,
                            [
                                *IDENTITY,
                                "candidate_month",
                                "execution_gross_ev_12h",
                                "execution_cost_return",
                                NET,
                            ],
                        ].copy()
                        book["score_name"] = score_name
                        book["mode"] = mode
                        book["score_value"] = selected[score_column].to_numpy(float)
                        books.append(book)
                        cutoff = float(selected[score_column].iloc[-1])
                        tie_rows = int(
                            np.isclose(
                                month_rows[score_column].to_numpy(float),
                                cutoff,
                                rtol=0.0,
                                atol=1e-14,
                            ).sum()
                        )
                        ties.append(
                            {
                                "month": str(month),
                                "score_name": score_name,
                                "mode": mode,
                                "cutoff": cutoff,
                                "cutoff_tie_rows": tie_rows,
                                "raw_secondary_used": bool(secondary),
                                "distinct_scores": int(
                                    month_rows[score_column].nunique()
                                ),
                            }
                        )
                metrics.append(record)
    books_frame = pd.concat(books, ignore_index=True)
    side_rows: list[dict[str, Any]] = []
    for (month, score, mode), local in books_frame.groupby(
        ["candidate_month", "score_name", "mode"], sort=True
    ):
        for side in ("long", "short"):
            cohort = local.loc[local["side_name"].eq(side)]
            side_rows.append(
                {
                    "month": str(month),
                    "score_name": score,
                    "mode": mode,
                    "side_name": side,
                    "rows": len(cohort),
                    "share": float(len(cohort) / len(local)),
                    "conditional_net_bps": (
                        float(cohort[NET].mean() * 1e4)
                        if len(cohort)
                        else np.nan
                    ),
                    "contribution_net_bps": (
                        float(cohort[NET].sum() * 1e4 / len(local))
                        if len(cohort)
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(metrics), books_frame, pd.DataFrame(ties).merge(
        pd.DataFrame(side_rows).groupby(
            ["month", "score_name", "mode"], as_index=False
        )["rows"].sum(),
        on=["month", "score_name", "mode"],
        validate="one_to_one",
    )


def _side_attribution(books: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, score, mode), local in books.groupby(
        ["candidate_month", "score_name", "mode"], sort=True
    ):
        for side in ("long", "short"):
            cohort = local.loc[local["side_name"].eq(side)]
            rows.append(
                {
                    "month": str(month),
                    "score_name": score,
                    "mode": mode,
                    "side_name": side,
                    "rows": len(cohort),
                    "share": float(len(cohort) / len(local)),
                    "conditional_net_bps": (
                        float(cohort[NET].mean() * 1e4)
                        if len(cohort)
                        else np.nan
                    ),
                    "contribution_net_bps": (
                        float(cohort[NET].sum() * 1e4 / len(local))
                        if len(cohort)
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(
    *,
    residual_root: Path,
    waterfall_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    frame, evidence = _load(residual_root, waterfall_root)
    mapped, audit = _map_scores(frame)
    metrics, books, ties = _evaluate(mapped)
    side = _side_attribution(books)
    stage = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir(parents=True)
    try:
        tables = {
            "mapped_candidates.parquet": mapped,
            "mapping_audit.parquet": audit,
            "period_metrics.parquet": metrics,
            "selection_books.parquet": books,
            "cutoff_ties.parquet": ties,
            "global_book_side_attribution.parquet": side,
        }
        outputs: dict[str, Any] = {}
        for name, table in tables.items():
            path = stage / name
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {
                "path": str((output_dir / name).resolve()),
                "rows": len(table),
                "sha256": sha256(path),
            }
        common = mapped.loc[mapped["common_mapping_eligible"]]
        if not common["execution_label_end_utc"].lt(
            common["execution_decision_utc"].dt.floor("D")
        ).all():
            # Current rows need not be resolved before their own snapshot; only
            # reference rows do.  The helper's audit and construction enforce
            # that distinction.  This branch is intentionally informational.
            current_rows_resolve_later = True
        else:
            current_rows_resolve_later = False
        manifest = {
            "schema": SCHEMA,
            "status": (
                "SEALED_FROZEN_SCORE_CAUSAL_RECENT_EV_MAPPING_DIAGNOSTIC_"
                "NO_PROMOTION_NO_REPLAY"
            ),
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": evidence,
            "contract": {
                "scores": "frozen exact-H12 v2 OOF score arms; no model refit",
                "mapping": (
                    "21 UTC days; exact-H12 label_end < daily snapshot; minimum "
                    "500 pooled rows; side isotonic shrunk toward pooled with "
                    "500-row support target"
                ),
                "selection": (
                    "one pooled-global top-k per month after mapped EV; raw score "
                    "then candidate ID only resolve isotonic plateaus; no quotas"
                ),
                "costs": (
                    "spread-inclusive gross minus one explicit round-trip cost "
                    "equals exact-H12 net"
                ),
                "actions": "timing, MAE, target-price and wait layers excluded",
                "warmup": (
                    "all score arms evaluated only on their exact common mapping-"
                    "eligible rows; unavailable warmup is retained in mapped output"
                ),
                "current_rows_resolve_after_own_snapshot_expected": (
                    current_rows_resolve_later
                ),
            },
            "constants": {
                "window_days": WINDOW_DAYS,
                "minimum_reference_rows": MIN_REFERENCE_ROWS,
                "side_support_target": SIDE_SUPPORT_TARGET,
            },
            "rows": {
                "source": len(mapped),
                "common_mapping_eligible": int(
                    mapped["common_mapping_eligible"].sum()
                ),
                "warmup_unmapped": int(
                    (~mapped["common_mapping_eligible"]).sum()
                ),
            },
            "score_registry": dict(SCORES),
            "outputs": outputs,
            "runner": binding(Path(__file__)),
        }
        write_json(stage / "manifest.json", manifest)
        manifest_digest = sha256(stage / "manifest.json")
        (stage / "manifest.sha256").write_text(
            f"{manifest_digest}  manifest.json\n", encoding="utf-8"
        )
        write_json(
            stage / "seal.json",
            {
                "schema": SCHEMA,
                "manifest_sha256": manifest_digest,
                "files_sha256": {
                    path.relative_to(stage).as_posix(): sha256(path)
                    for path in sorted(stage.rglob("*"))
                    if path.is_file() and path.name != "seal.json"
                },
            },
        )
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL)
    value.add_argument("--waterfall-root", type=Path, default=DEFAULT_WATERFALL)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    print(json.dumps(safe(run(**vars(parser().parse_args()))), sort_keys=True))
