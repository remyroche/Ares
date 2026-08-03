#!/usr/bin/env python3
"""Materialize the July 20--23 exact all-score retrospective bridge.

This joins the frozen Pack-B, pre-entry, raw direct/capture, direct-quantile,
transfer-adapter and exact 12-hour policy-label sources on all four identity
fields.  Every output is stamped retrospective/non-promotable/not-OOS.  The
bridge contains raw scores only; the existing recent-EV mapped policy remains
a separate evidence source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.materialize_source_separated_ic_ev_waterfall import (
    IDENTITY_COLUMNS,
    cutoff_ties,
    full_ic,
    response_20bin,
    safe,
    score_columns,
    score_compression,
    sha256,
    tail_metrics,
    validate_source,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RETRO_ROOT = ROOT / (
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
)
DEFAULT_PACKB = RETRO_ROOT / "packb/packb_forward_context.parquet"
DEFAULT_PACKB_MANIFEST = DEFAULT_PACKB.with_name("manifest.json")
DEFAULT_PREENTRY = RETRO_ROOT / "preentry/preentry.parquet"
DEFAULT_PREENTRY_MANIFEST = DEFAULT_PREENTRY.with_name("manifest.json")
DEFAULT_SCORED = RETRO_ROOT / "scored/scored_population.parquet"
DEFAULT_SCORED_MANIFEST = DEFAULT_SCORED.with_name("manifest.json")
DEFAULT_LABELS = RETRO_ROOT / "labels_12h/execution_ev_policy_labels.parquet"
DEFAULT_LABELS_MANIFEST = DEFAULT_LABELS.with_name("manifest.json")
DEFAULT_DIRECT = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/"
    "current_predictions_before_outcomes.parquet"
)
DEFAULT_DIRECT_MANIFEST = DEFAULT_DIRECT.with_name("manifest.json")
DEFAULT_ADAPTER = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_transfer_adapter_current_score_20260730_v2/"
    "current_predictions_before_outcomes.parquet"
)
DEFAULT_ADAPTER_MANIFEST = DEFAULT_ADAPTER.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/july20_23_retrospective_allscore_bridge_20260730_v1"
)

EXPECTED_ROWS = 5_760
SOURCE_FAMILY = "july20_23_retrospective_exact12h_raw_allscore"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{source} has duplicate four-field identities")
    if result["candidate_id"].duplicated().any():
        raise ValueError(f"{source} has duplicate candidate IDs")
    return result


def _require_output(
    manifest_path: Path,
    path: Path,
    *,
    schema: str,
    output_key: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != schema:
        raise ValueError(f"unexpected schema at {manifest_path}")
    record = (
        manifest.get("outputs", {}).get(output_key, {})
        if output_key
        else manifest.get("output", {})
    )
    declared = Path(str(record.get("path")))
    if not declared.is_absolute():
        declared = ROOT / declared
    if declared.resolve() != path.resolve():
        raise ValueError(f"manifest path mismatch for {path}")
    if str(record.get("sha256")) != sha256(path):
        raise ValueError(f"manifest hash mismatch for {path}")
    return manifest


def _exact_join(
    anchor: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    source = _identity(source, source_name)
    if len(source) != len(anchor):
        raise ValueError(f"{source_name} rows differ from anchor")
    missing = sorted(set(columns).difference(source.columns))
    if missing:
        raise ValueError(f"{source_name} missing fields: {missing}")
    joined = anchor.merge(
        source.loc[:, [*IDENTITY_COLUMNS, *columns]],
        on=list(IDENTITY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            f"{source_name} identity coverage failed: "
            f"{joined['_merge'].value_counts().to_dict()}"
        )
    return joined.drop(columns="_merge")


def _check_available(
    frame: pd.DataFrame, columns: Sequence[str], decision_column: str
) -> None:
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    for column in columns:
        available = pd.to_datetime(frame[column], utc=True, errors="raise")
        if not available.le(decision).all():
            raise ValueError(f"{column} is not available at decision")


def build_bridge(
    labels: pd.DataFrame,
    packb: pd.DataFrame,
    preentry: pd.DataFrame,
    scored: pd.DataFrame,
    direct: pd.DataFrame,
    adapter: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = _identity(labels, "exact 12h labels")
    if len(anchor) != expected_rows:
        raise ValueError(f"label rows {len(anchor)} != expected {expected_rows}")
    anchor["execution_decision_utc"] = pd.to_datetime(
        anchor["execution_decision_utc"], utc=True, errors="raise"
    )
    anchor["execution_label_end_utc"] = pd.to_datetime(
        anchor["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not anchor["execution_label_end_utc"].eq(
        anchor["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("labels are not on the exact 12h horizon")

    packb_columns = [
        "base_oof_score",
        "base_alpha_ev",
        "residual_delta_ev",
        "existing_alpha_ev",
        "base_available_at",
        "residual_available_at",
    ]
    frame = _exact_join(
        anchor, packb, source_name="Pack-B", columns=packb_columns
    )
    _check_available(
        frame, ["base_available_at", "residual_available_at"], "execution_decision_utc"
    )

    preentry = preentry.rename(
        columns={"existing_alpha_ev": "preentry_existing_alpha_ev"}
    )
    preentry_columns = [
        "preentry_existing_alpha_ev",
        "pred_peak_MFE_12h_ATR",
        "oof_clean_favorable_probability",
        "feature_available_at",
        "base_available_at",
        "residual_available_at",
        "peak_mfe_available_at",
        "path_catboost_available_at",
        "clean_probability_available_at",
    ]
    frame = _exact_join(
        frame, preentry, source_name="pre-entry", columns=preentry_columns
    )
    _check_available(
        frame,
        [
            "feature_available_at",
            "base_available_at_y",
            "residual_available_at_y",
            "peak_mfe_available_at",
            "path_catboost_available_at",
            "clean_probability_available_at",
        ],
        "execution_decision_utc",
    )
    if not np.array_equal(
        pd.to_numeric(frame["existing_alpha_ev"], errors="raise").to_numpy(float),
        pd.to_numeric(
            frame["preentry_existing_alpha_ev"], errors="raise"
        ).to_numpy(float),
    ):
        raise ValueError("pre-entry alpha score differs from Pack-B")

    scored = scored.rename(
        columns={"existing_alpha_ev": "scored_existing_alpha_ev"}
    )
    scored_columns = [
        "scored_existing_alpha_ev",
        "final_direct_net_raw",
        "final_capture_probability",
        "frozen_margin_capture_interaction_raw",
        "direct_ev_available_at",
        "capture_probability_available_at",
        "mapping_available_at",
    ]
    frame = _exact_join(
        frame, scored, source_name="retrospective scored", columns=scored_columns
    )
    _check_available(
        frame,
        [
            "direct_ev_available_at",
            "capture_probability_available_at",
            "mapping_available_at",
        ],
        "execution_decision_utc",
    )
    if not np.array_equal(
        pd.to_numeric(frame["existing_alpha_ev"], errors="raise").to_numpy(float),
        pd.to_numeric(frame["scored_existing_alpha_ev"], errors="raise").to_numpy(
            float
        ),
    ):
        raise ValueError("scored alpha differs from Pack-B")

    direct = direct.rename(
        columns={
            "q25_net_bps": "challenger_q25_net_bps",
            "q50_net_bps": "challenger_q50_net_bps",
        }
    )
    frame = _exact_join(
        frame,
        direct,
        source_name="current direct q",
        columns=["challenger_q25_net_bps", "challenger_q50_net_bps"],
    )

    adapter = adapter.rename(
        columns={
            "q25_net_bps": "adapter_source_q25_net_bps",
            "q50_net_bps": "adapter_source_q50_net_bps",
            "base_oof_score": "adapter_base_oof_score",
        }
    )
    frame = _exact_join(
        frame,
        adapter,
        source_name="current transfer adapter",
        columns=[
            "adapter_source_q25_net_bps",
            "adapter_source_q50_net_bps",
            "adapter_base_oof_score",
            "score_parent_bps",
            "score_adapter_bps",
            "score_reliability_bps",
            "score_adapter_reliability_bps",
        ],
    )
    if not np.array_equal(
        pd.to_numeric(frame["base_oof_score"], errors="raise").to_numpy(float),
        pd.to_numeric(frame["adapter_base_oof_score"], errors="raise").to_numpy(
            float
        ),
    ):
        raise ValueError("adapter base score differs from exact Pack-B join")

    score_map = {
        "score_base_alpha": "base_oof_score",
        "score_base_alpha_ev": "base_alpha_ev",
        "score_residual_delta_ev": "residual_delta_ev",
        "score_existing_alpha_ev": "existing_alpha_ev",
        "score_preentry_peak_mfe_atr": "pred_peak_MFE_12h_ATR",
        "score_preentry_clean_favorable_probability": "oof_clean_favorable_probability",
        "score_final_direct_net_raw": "final_direct_net_raw",
        "score_final_capture_probability": "final_capture_probability",
        "score_margin_capture_interaction_raw": "frozen_margin_capture_interaction_raw",
        "score_direct_q25_challenger_bps": "challenger_q25_net_bps",
        "score_direct_q50_challenger_bps": "challenger_q50_net_bps",
        "score_transfer_source_q25_bps": "adapter_source_q25_net_bps",
        "score_transfer_source_q50_bps": "adapter_source_q50_net_bps",
        "score_transfer_parent_bps": "score_parent_bps",
        "score_transfer_adapter_bps": "score_adapter_bps",
        "score_transfer_reliability_bps": "score_reliability_bps",
        "score_transfer_adapter_reliability_bps": "score_adapter_reliability_bps",
    }
    registry_rows: list[dict[str, Any]] = []
    for output, source in score_map.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"non-finite raw score: {output}")
        frame[output] = values
        registry_rows.append(
            {
                "score": output,
                "source_column": source,
                "retrospective_nonpromotable_not_oos": True,
                "mapped": False,
            }
        )

    keep = [
        *IDENTITY_COLUMNS,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_reason",
        "execution_exit_hour",
        *score_map,
    ]
    frame = frame.loc[:, keep].copy()
    frame["candidate_month"] = "2026-07"
    frame["candidate_day"] = frame["__ts__"].dt.strftime("%Y-%m-%d")
    frame["opportunity_gross_above_cost_0bps"] = frame[
        "execution_net_ev_12h"
    ].gt(0.0)
    frame["source_family"] = SOURCE_FAMILY
    frame["evidence_status"] = "retrospective_nonpromotable_not_oos"
    if any("mapped" in column.lower() for column in frame.columns):
        raise ValueError("mapped fields are forbidden in raw retrospective bridge")
    validate_source(frame, {"source_family": SOURCE_FAMILY})
    return (
        frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
            drop=True
        ),
        pd.DataFrame(registry_rows),
    )


def _diagnostics(
    frame: pd.DataFrame, *, day_level: bool = False
) -> dict[str, pd.DataFrame]:
    source = frame
    if day_level:
        source = frame.copy()
        source["candidate_month"] = source["candidate_day"]
    parts: dict[str, list[pd.DataFrame]] = {
        key: []
        for key in (
            "full_ic",
            "tails",
            "compression",
            "response_cells",
            "response_summary",
            "cutoff_ties",
        )
    }
    for score in score_columns(source):
        parts["full_ic"].append(full_ic(source, source_family=SOURCE_FAMILY, score=score))
        parts["tails"].append(tail_metrics(source, source_family=SOURCE_FAMILY, score=score))
        parts["compression"].append(
            score_compression(source, source_family=SOURCE_FAMILY, score=score)
        )
        cells, summary = response_20bin(source, source_family=SOURCE_FAMILY, score=score)
        parts["response_cells"].append(cells)
        parts["response_summary"].append(summary)
        parts["cutoff_ties"].append(
            cutoff_ties(source, source_family=SOURCE_FAMILY, score=score)
        )
    prefix = "daily_" if day_level else ""
    return {
        f"{prefix}{name}": pd.concat(values, ignore_index=True)
        for name, values in parts.items()
    }


def run(
    packb_path: Path,
    packb_manifest_path: Path,
    preentry_path: Path,
    preentry_manifest_path: Path,
    scored_path: Path,
    scored_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    direct_path: Path,
    direct_manifest_path: Path,
    adapter_path: Path,
    adapter_manifest_path: Path,
    output_dir: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    packb_manifest = _require_output(
        packb_manifest_path,
        packb_path,
        schema="packb_final_refits_forward_v1",
    )
    preentry_manifest = _require_output(
        preentry_manifest_path,
        preentry_path,
        schema="execution_ev_forward_preentry_v1",
    )
    scored_manifest = _require_output(
        scored_manifest_path,
        scored_path,
        schema="execution_ev_retrospective_scored_population_v1",
        output_key="scored_population",
    )
    labels_manifest = _require_output(
        labels_manifest_path,
        labels_path,
        schema="execution_ev_deployed_policy_1m_labels_v1",
    )
    direct_manifest = _require_output(
        direct_manifest_path,
        direct_path,
        schema="cross_era_direct_net_quantile_challenger_v1",
        output_key="current_predictions_before_outcomes",
    )
    adapter_manifest = _require_output(
        adapter_manifest_path,
        adapter_path,
        schema="cross_era_direct_net_transfer_adapter_ablation_v1",
        output_key="predictions",
    )
    if labels_manifest.get("timing", {}).get("horizon_minutes") != 720:
        raise ValueError("July policy label manifest is not 720 minutes")
    if scored_manifest.get("promotion_eligible") is not False:
        raise ValueError("retrospective scored source promotion status changed")
    if adapter_manifest.get("status") != "scored_label_free_before_current_outcomes":
        raise ValueError("adapter source is not the label-free score artifact")

    frame, registry = build_bridge(
        pd.read_parquet(labels_path),
        pd.read_parquet(packb_path),
        pd.read_parquet(preentry_path),
        pd.read_parquet(scored_path),
        pd.read_parquet(
            direct_path,
            columns=[
                *IDENTITY_COLUMNS,
                "q25_net_bps",
                "q50_net_bps",
            ],
        ),
        pd.read_parquet(adapter_path),
        expected_rows=expected_rows,
    )
    diagnostics = {**_diagnostics(frame), **_diagnostics(frame, day_level=True)}
    coverage = (
        frame.groupby(["candidate_day", "side_name"], sort=True, observed=True)
        .agg(rows=("candidate_id", "size"))
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "retrospective_allscore_bridge": frame,
        "score_registry": registry,
        "day_side_coverage": coverage,
        **diagnostics,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name, result in frames.items():
        path = output_dir / f"{name}.parquet"
        result.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(result)),
        }

    input_paths = {
        "packb": (packb_path, packb_manifest_path),
        "preentry": (preentry_path, preentry_manifest_path),
        "scored": (scored_path, scored_manifest_path),
        "labels": (labels_path, labels_manifest_path),
        "direct": (direct_path, direct_manifest_path),
        "adapter": (adapter_path, adapter_manifest_path),
    }
    report = {
        "schema": "july20_23_retrospective_allscore_bridge_v1",
        "status": "RETROSPECTIVE_NONPROMOTABLE_NOT_OOS_RAW_SCORES_ONLY",
        "rows": int(len(frame)),
        "period": {
            "signal_start": frame["__ts__"].min().isoformat(),
            "signal_end": frame["__ts__"].max().isoformat(),
            "label_end": frame["execution_label_end_utc"].max().isoformat(),
        },
        "contracts": {
            "identity": list(IDENTITY_COLUMNS),
            "coverage": "5760 exact rows, 2880 per side, no subset or imputation",
            "availability": (
                "all persisted core score availability timestamps <= decision; "
                "direct-q and adapter files lack availability timestamps and "
                "inherit decision lineage only through exact hash-bound Pack-B join"
            ),
            "label_horizon": "exact 1m decision+12h policy replay",
            "selection": (
                "raw score only; pooled global top 1/5/10/20 primary; "
                "side/day diagnostics; candidate-ID tie break"
            ),
            "mapping": (
                "mapped_execution_ev and global admission fields are excluded; "
                "the frozen mapped policy is a separate existing evidence source"
            ),
            "mfe_semantics": "MFE is an upper-bound ceiling, not attainable gross",
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "manifest_path": str(manifest),
                "manifest_sha256": sha256(manifest),
            }
            for name, (path, manifest) in input_paths.items()
        },
        "score_contract": {
            "declared_scores": score_columns(frame),
            "mapped_score_forbidden": True,
            "mapped_columns_emitted": [],
        },
        "outputs": outputs,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "promotion_eligible": False,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, report)
    (output_dir / "manifest.sha256").write_text(
        sha256(manifest_path) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packb", type=Path, default=DEFAULT_PACKB)
    parser.add_argument("--packb-manifest", type=Path, default=DEFAULT_PACKB_MANIFEST)
    parser.add_argument("--preentry", type=Path, default=DEFAULT_PREENTRY)
    parser.add_argument(
        "--preentry-manifest", type=Path, default=DEFAULT_PREENTRY_MANIFEST
    )
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--scored-manifest", type=Path, default=DEFAULT_SCORED_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--labels-manifest", type=Path, default=DEFAULT_LABELS_MANIFEST)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--direct-manifest", type=Path, default=DEFAULT_DIRECT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--adapter-manifest", type=Path, default=DEFAULT_ADAPTER_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            safe(
                run(
                    args.packb,
                    args.packb_manifest,
                    args.preentry,
                    args.preentry_manifest,
                    args.scored,
                    args.scored_manifest,
                    args.labels,
                    args.labels_manifest,
                    args.direct,
                    args.direct_manifest,
                    args.adapter,
                    args.adapter_manifest,
                    args.output_dir,
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
