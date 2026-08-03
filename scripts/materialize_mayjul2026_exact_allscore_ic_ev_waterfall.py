#!/usr/bin/env python3
"""Materialize the exact May--July 10, 2026 all-score IC-to-EV waterfall.

The anchor is the signed 720-minute, exact-one-minute policy ledger.  Strict
base and residual OOF streams join on their native four-field identity.  The
direct-q and transfer-adapter sources use an older underscore symbol encoding;
this runner permits that source-local repair only after proving that symbol,
timestamp, timeframe and side all agree with the candidate ID.  No mapped
score is read or emitted and the output is diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.materialize_source_separated_ic_ev_waterfall import (
    IDENTITY_COLUMNS,
    cutoff_ties,
    fixed_composition,
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
DEFAULT_LABELS = ROOT / (
    "data_perp/artifacts/execution_ev_policy_labels_12h_20260725_v1/"
    "execution_ev_policy_labels.parquet"
)
DEFAULT_LABELS_MANIFEST = DEFAULT_LABELS.with_name("manifest.json")
DEFAULT_BASE = ROOT / (
    "data_perp/artifacts/packb_side_local_top40_july20_20260726_v1_31_8/"
    "base_candidate_population.parquet"
)
DEFAULT_BASE_MANIFEST = DEFAULT_BASE.with_name("manifest.json")
DEFAULT_RESIDUAL = ROOT / (
    "data_perp/artifacts/packb_side_local_residual_oof_july20_20260726_v1_31_8/"
    "oof_predictions.parquet"
)
DEFAULT_RESIDUAL_MANIFEST = DEFAULT_RESIDUAL.with_name("manifest.json")
DEFAULT_DIRECT = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/"
    "historical_oof_winner.parquet"
)
DEFAULT_DIRECT_MANIFEST = DEFAULT_DIRECT.with_name("manifest.json")
DEFAULT_ADAPTER = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_transfer_adapter_ablation_20260730_v2/"
    "historical_oof_all_arms.parquet"
)
DEFAULT_ADAPTER_MANIFEST = DEFAULT_ADAPTER.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
)

EXPECTED_ROWS = 127_777
SOURCE_FAMILY = "mayjul2026_exact12h_allscore_oof"
SCORE_SOURCES = {
    "score_base_alpha": ("base", "prediction"),
    "score_base_expected_ev": ("residual", "base_expected_ev"),
    "score_residual_delta_ev": ("residual", "residual_delta_ev"),
    "score_residual_expected_ev": ("residual", "residual_expected_ev"),
    "score_direct_q25_challenger_bps": ("direct", "q25_net_bps"),
    "score_direct_q50_challenger_bps": ("direct", "q50_net_bps"),
    "score_transfer_source_q25_bps": ("adapter", "q25_net_bps"),
    "score_transfer_source_q50_bps": ("adapter", "q50_net_bps"),
    "score_transfer_parent_bps": ("adapter", "score_parent_bps"),
    "score_transfer_adapter_bps": ("adapter", "score_adapter_bps"),
    "score_transfer_reliability_bps": ("adapter", "score_reliability_bps"),
    "score_transfer_adapter_reliability_bps": (
        "adapter",
        "score_adapter_reliability_bps",
    ),
}


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


def _manifest_output(
    manifest_path: Path,
    *,
    schema: str,
    path: Path,
    output_key: str | None = None,
    direct_hash_key: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != schema:
        raise ValueError(f"unexpected manifest schema at {manifest_path}")
    if output_key is not None:
        record = manifest.get("outputs", {}).get(output_key, {})
        declared_path = Path(str(record.get("path")))
        declared_hash = str(record.get("sha256"))
    elif direct_hash_key is not None:
        declared_path = path
        declared_hash = str(manifest.get(direct_hash_key))
    else:
        record = manifest.get("output", {})
        declared_path = Path(str(record.get("path")))
        declared_hash = str(record.get("sha256"))
    if not declared_path.is_absolute():
        declared_path = ROOT / declared_path
    if declared_path.resolve() != path.resolve():
        raise ValueError(f"manifest path mismatch for {path}")
    if declared_hash != sha256(path):
        raise ValueError(f"manifest hash mismatch for {path}")
    return manifest


def _canonicalize_direct_identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Repair only the source-local underscore quote separator, fail closed."""

    result = _identity(frame, source)
    pieces = result["candidate_id"].str.split("|", expand=True)
    if pieces.shape[1] != 4:
        raise ValueError(f"{source} candidate ID does not have four tokens")
    candidate_symbol = pieces[0].astype(str)
    candidate_ts = pd.to_datetime(pieces[1], utc=True, errors="raise")
    candidate_timeframe = pieces[2].astype(str)
    candidate_side = pieces[3].astype(str)
    encoded_symbol = candidate_symbol.str.replace("/", "_", regex=False)
    if not encoded_symbol.equals(result["__symbol__"]):
        raise ValueError(f"{source} symbol encoding disagrees with candidate ID")
    if not candidate_ts.equals(result["__ts__"]):
        raise ValueError(f"{source} timestamp disagrees with candidate ID")
    if not candidate_timeframe.eq("1h").all():
        raise ValueError(f"{source} candidate timeframe is not 1h")
    if not candidate_side.equals(result["side_name"]):
        raise ValueError(f"{source} side disagrees with candidate ID")
    result["__symbol__"] = candidate_symbol
    return result


def _left_exact(
    anchor: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(set(columns).difference(source.columns))
    if missing:
        raise ValueError(f"{source_name} missing fields: {missing}")
    selected = source.loc[:, [*IDENTITY_COLUMNS, *columns]].copy()
    joined = anchor.merge(
        selected,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            f"{source_name} exact identity coverage failed: "
            f"{joined['_merge'].value_counts().to_dict()}"
        )
    return joined.drop(columns="_merge")


def _check_base_oof(frame: pd.DataFrame) -> None:
    if not frame["prediction_source"].eq("outer_oof_fold_model").all():
        raise ValueError("base prediction is not uniformly outer OOF")
    if not frame["base_fold_fit_scope"].eq(
        "strict_prior_resolved_labels_side_local"
    ).all():
        raise ValueError("base fold fit scope is not strict prior-resolved OOF")
    validation = pd.to_datetime(frame["validation_start"], utc=True, errors="raise")
    cutoff = pd.to_datetime(
        frame["train_decision_cutoff"], utc=True, errors="raise"
    )
    resolved = pd.to_datetime(
        frame["label_resolution_available_at"], utc=True, errors="raise"
    )
    if not cutoff.lt(validation).all():
        raise ValueError("base train cutoff reaches validation")
    if not resolved.le(cutoff).all():
        raise ValueError("base training includes unresolved labels")


def _check_residual_oof(frame: pd.DataFrame) -> None:
    if not frame["residual_is_oof"].astype(bool).all():
        raise ValueError("residual stream is not uniformly OOF")
    validation = pd.to_datetime(
        frame["residual_validation_start"], utc=True, errors="raise"
    )
    cutoff = pd.to_datetime(
        frame["residual_train_decision_cutoff"], utc=True, errors="raise"
    )
    available = pd.to_datetime(
        frame["residual_prediction_available_at"], utc=True, errors="raise"
    )
    decision = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    if not cutoff.lt(validation).all():
        raise ValueError("residual train cutoff reaches validation")
    if not available.le(decision).all():
        raise ValueError("residual prediction is unavailable at decision")


def build_allscore_frame(
    labels: pd.DataFrame,
    base: pd.DataFrame,
    residual: pd.DataFrame,
    direct: pd.DataFrame,
    adapter: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact all-score population and score registry."""

    anchor = _identity(labels, "exact 12h policy labels")
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
        raise ValueError("policy labels are not on the exact 12h after-decision horizon")

    base = _identity(base, "base OOF source")
    residual = _identity(residual, "residual OOF source")
    direct = _canonicalize_direct_identity(direct, "direct q OOF source")
    adapter = _canonicalize_direct_identity(adapter, "transfer adapter OOF source")

    base_columns = [
        "prediction",
        "__first_touch_target_soft__",
        "prediction_source",
        "base_fold_fit_scope",
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "oos_fold",
    ]
    frame = _left_exact(anchor, base, source_name="base", columns=base_columns)
    _check_base_oof(frame)

    residual_columns = [
        "base_expected_ev",
        "residual_delta_ev",
        "residual_expected_ev",
        "residual_is_oof",
        "residual_validation_start",
        "residual_train_decision_cutoff",
        "residual_prediction_available_at",
        "residual_oof_fold",
        "__label_resolution_ts__",
    ]
    frame = _left_exact(
        frame, residual, source_name="residual", columns=residual_columns
    )
    _check_residual_oof(frame)
    residual_resolution = pd.to_datetime(
        frame["__label_resolution_ts__"], utc=True, errors="raise"
    )
    residual_horizon_delta = (
        residual_resolution - frame["execution_label_end_utc"]
    )
    if not residual_horizon_delta.eq(pd.Timedelta(hours=12)).all():
        raise ValueError(
            "residual legacy target is not uniformly 12h longer than exact anchor"
        )

    direct_columns = [
        "q25_net_bps",
        "q50_net_bps",
        "execution_net_ev_12h",
        "label_resolution_utc",
    ]
    direct = direct.rename(
        columns={
            "q25_net_bps": "direct_q25_net_bps",
            "q50_net_bps": "direct_q50_net_bps",
            "execution_net_ev_12h": "direct_execution_net_ev_12h",
            "label_resolution_utc": "direct_label_resolution_utc",
        }
    )
    frame = _left_exact(
        frame,
        direct,
        source_name="direct q",
        columns=[
            "direct_q25_net_bps",
            "direct_q50_net_bps",
            "direct_execution_net_ev_12h",
            "direct_label_resolution_utc",
        ],
    )

    adapter = adapter.rename(
        columns={
            "q25_net_bps": "transfer_q25_net_bps",
            "q50_net_bps": "transfer_q50_net_bps",
            "execution_net_ev_12h": "transfer_execution_net_ev_12h",
            "label_resolution_utc": "transfer_label_resolution_utc",
        }
    )
    frame = _left_exact(
        frame,
        adapter,
        source_name="transfer adapter",
        columns=[
            "transfer_q25_net_bps",
            "transfer_q50_net_bps",
            "score_parent_bps",
            "score_adapter_bps",
            "score_reliability_bps",
            "score_adapter_reliability_bps",
            "transfer_execution_net_ev_12h",
            "transfer_label_resolution_utc",
            "fold",
        ],
    )

    anchor_net = pd.to_numeric(
        frame["execution_net_ev_12h"], errors="raise"
    ).to_numpy(float)
    for source_name, net_column, resolution_column in (
        (
            "direct q",
            "direct_execution_net_ev_12h",
            "direct_label_resolution_utc",
        ),
        (
            "transfer adapter",
            "transfer_execution_net_ev_12h",
            "transfer_label_resolution_utc",
        ),
    ):
        source_net = pd.to_numeric(frame[net_column], errors="raise").to_numpy(float)
        if not np.array_equal(anchor_net, source_net):
            raise ValueError(f"{source_name} realized net differs from exact anchor")
        source_resolution = pd.to_datetime(
            frame[resolution_column], utc=True, errors="raise"
        )
        if not source_resolution.equals(frame["execution_label_end_utc"]):
            raise ValueError(f"{source_name} label horizon differs from exact anchor")

    source_lookup: dict[tuple[str, str], str] = {
        ("base", "prediction"): "prediction",
        ("residual", "base_expected_ev"): "base_expected_ev",
        ("residual", "residual_delta_ev"): "residual_delta_ev",
        ("residual", "residual_expected_ev"): "residual_expected_ev",
        ("direct", "q25_net_bps"): "direct_q25_net_bps",
        ("direct", "q50_net_bps"): "direct_q50_net_bps",
        ("adapter", "q25_net_bps"): "transfer_q25_net_bps",
        ("adapter", "q50_net_bps"): "transfer_q50_net_bps",
        ("adapter", "score_parent_bps"): "score_parent_bps",
        ("adapter", "score_adapter_bps"): "score_adapter_bps",
        ("adapter", "score_reliability_bps"): "score_reliability_bps",
        (
            "adapter",
            "score_adapter_reliability_bps",
        ): "score_adapter_reliability_bps",
    }
    registry_rows: list[dict[str, Any]] = []
    for output_score, (source_family, source_column) in SCORE_SOURCES.items():
        joined_column = source_lookup[(source_family, source_column)]
        values = pd.to_numeric(frame[joined_column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"non-finite declared score: {output_score}")
        frame[output_score] = values
        registry_rows.append(
            {
                "score": output_score,
                "source_family": source_family,
                "source_column": source_column,
                "units": "basis_points" if output_score.endswith("_bps") else "native",
                "strict_oof": True,
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
        "__first_touch_target_soft__",
        *SCORE_SOURCES,
    ]
    frame = frame.loc[:, keep].copy()
    frame["candidate_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    frame["opportunity_gross_above_cost_0bps"] = frame[
        "execution_net_ev_12h"
    ].gt(0.0)
    frame["source_family"] = SOURCE_FAMILY
    if any("mapped" in column.lower() for column in frame.columns):
        raise ValueError("mapped fields are forbidden in all-score waterfall")
    validate_source(frame, {"source_family": SOURCE_FAMILY})
    registry = pd.DataFrame(registry_rows)
    return (
        frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
            drop=True
        ),
        registry,
    )


def _emit_diagnostics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {
        key: []
        for key in (
            "full_ic",
            "tails",
            "compression",
            "response_cells",
            "response_summary",
            "cutoff_ties",
            "fixed_composition",
        )
    }
    for score in score_columns(frame):
        parts["full_ic"].append(full_ic(frame, source_family=SOURCE_FAMILY, score=score))
        parts["tails"].append(tail_metrics(frame, source_family=SOURCE_FAMILY, score=score))
        parts["compression"].append(
            score_compression(frame, source_family=SOURCE_FAMILY, score=score)
        )
        cells, summary = response_20bin(frame, source_family=SOURCE_FAMILY, score=score)
        parts["response_cells"].append(cells)
        parts["response_summary"].append(summary)
        parts["cutoff_ties"].append(
            cutoff_ties(frame, source_family=SOURCE_FAMILY, score=score)
        )
        parts["fixed_composition"].append(
            fixed_composition(frame, source_family=SOURCE_FAMILY, score=score)
        )
    return {
        name: pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        for name, values in parts.items()
    }


def run(
    labels_path: Path,
    labels_manifest_path: Path,
    base_path: Path,
    base_manifest_path: Path,
    residual_path: Path,
    residual_manifest_path: Path,
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
    manifests = {
        "labels": _manifest_output(
            labels_manifest_path,
            schema="execution_ev_deployed_policy_1m_labels_v1",
            path=labels_path,
        ),
        "base": _manifest_output(
            base_manifest_path,
            schema="base_candidate_population_v2",
            path=base_path,
        ),
        "residual": _manifest_output(
            residual_manifest_path,
            schema="packb_side_local_residual_oof_v1",
            path=residual_path,
            direct_hash_key="oof_predictions_sha256",
        ),
        "direct": _manifest_output(
            direct_manifest_path,
            schema="cross_era_direct_net_quantile_challenger_v1",
            path=direct_path,
            output_key="historical_oof_winner",
        ),
        "adapter": _manifest_output(
            adapter_manifest_path,
            schema="cross_era_direct_net_transfer_adapter_ablation_v1",
            path=adapter_path,
            output_key="historical_oof_all_arms",
        ),
    }
    if manifests["labels"].get("timing", {}).get("horizon_minutes") != 720:
        raise ValueError("exact policy manifest is not the 720-minute horizon")
    if manifests["direct"].get("promotion_eligible") is not False:
        raise ValueError("direct challenger research status changed unexpectedly")

    frame, registry = build_allscore_frame(
        pd.read_parquet(labels_path),
        pd.read_parquet(base_path),
        pd.read_parquet(residual_path),
        pd.read_parquet(
            direct_path,
            columns=[
                *IDENTITY_COLUMNS,
                "q25_net_bps",
                "q50_net_bps",
                "execution_net_ev_12h",
                "label_resolution_utc",
            ],
        ),
        pd.read_parquet(
            adapter_path,
            columns=[
                *IDENTITY_COLUMNS,
                "q25_net_bps",
                "q50_net_bps",
                "score_parent_bps",
                "score_adapter_bps",
                "score_reliability_bps",
                "score_adapter_reliability_bps",
                "execution_net_ev_12h",
                "label_resolution_utc",
                "fold",
            ],
        ),
        expected_rows=expected_rows,
    )
    diagnostics = _emit_diagnostics(frame)
    coverage = (
        frame.groupby(["candidate_month", "side_name"], sort=True, observed=True)
        .agg(
            rows=("candidate_id", "size"),
            first_signal_utc=("__ts__", "min"),
            last_signal_utc=("__ts__", "max"),
            last_label_end_utc=("execution_label_end_utc", "max"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    output_frames = {
        "allscore_waterfall": frame,
        "score_registry": registry,
        "month_side_coverage": coverage,
        **diagnostics,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name, result in output_frames.items():
        path = output_dir / f"{name}.parquet"
        result.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(result)),
        }

    input_paths = {
        "labels": (labels_path, labels_manifest_path),
        "base": (base_path, base_manifest_path),
        "residual": (residual_path, residual_manifest_path),
        "direct": (direct_path, direct_manifest_path),
        "adapter": (adapter_path, adapter_manifest_path),
    }
    report: dict[str, Any] = {
        "schema": "mayjul2026_exact_allscore_ic_ev_waterfall_v1",
        "status": "DIAGNOSTIC_ONLY_STRICT_OOF_NO_MAPPING_NO_PROMOTION",
        "rows": int(len(frame)),
        "period": {
            "signal_start": frame["__ts__"].min().isoformat(),
            "signal_end": frame["__ts__"].max().isoformat(),
            "label_end": frame["execution_label_end_utc"].max().isoformat(),
        },
        "contracts": {
            "identity": list(IDENTITY_COLUMNS),
            "source_local_symbol_repair": (
                "direct/adapter only: canonical symbol comes from candidate "
                "ID after asserting source symbol == canonical symbol with "
                "'/' encoded as '_', timestamp token == __ts__, timeframe "
                "== 1h and side token == side_name"
            ),
            "oof": (
                "base and residual strict prior-resolved side-local OOF; "
                "direct challenger and transfer arms historical OOF"
            ),
            "residual_target_semantics": (
                "residual scores were trained on the legacy fixed-1%-cost "
                "24h residual target; their per-candidate label-resolution "
                "timestamp is exactly 12h later than this waterfall's 12h "
                "execution endpoint. They are evaluated only as OOF score "
                "arms, never represented as same-target calibration."
            ),
            "label_horizon": (
                "signed 720-minute exact-1m policy replay; decision+12h "
                "label endpoint required on every row"
            ),
            "economics": (
                "gross includes executable spread; explicit 1% round-trip "
                "fee is deducted once; gross-cost=net"
            ),
            "selection": (
                "month-level pooled global top 1/5/10/20 is primary; "
                "side-local is attribution; raw score descending with "
                "candidate-ID tie break; no additional or recent-EV mapping"
            ),
            "mfe_semantics": "raw return MFE is a ceiling, not attainable gross",
            "direct_source_separation": (
                "q25 challenger and transfer-parent/adapter arms remain "
                "distinct because their raw scores are not identical"
            ),
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
            for name, (path, manifest_path) in input_paths.items()
        },
        "score_contract": {
            "declared_scores": score_columns(frame),
            "mapped_score_forbidden": True,
            "mapped_columns_emitted": [],
            "recent_ev_mapping_applied": False,
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
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--labels-manifest", type=Path, default=DEFAULT_LABELS_MANIFEST)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument(
        "--residual-manifest", type=Path, default=DEFAULT_RESIDUAL_MANIFEST
    )
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
                    args.labels,
                    args.labels_manifest,
                    args.base,
                    args.base_manifest,
                    args.residual,
                    args.residual_manifest,
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
