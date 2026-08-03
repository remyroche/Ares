#!/usr/bin/env python3
"""Repair full-base opportunity selection using raw OOF evidence only.

The source v1 model predictions are valid; its mapped-development selection is
not.  This runner hash-verifies and reuses those raw OOF predictions, selects
feature arms and geometry before fitting any mapper, fits only missing April
forward models, and then applies the canonical score-specific causal recent-EV
map.  April has already been inspected and is diagnostic only.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_canonical_full_base_opportunity_ablation as base
from scripts import run_short_winner_causal_recent_ev_mapping_v5 as mapping

SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260729_v1"
)
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260730_v2"
)
SCHEMA = "canonical_full_base_opportunity_ablation_raw_oof_repair_v2"
RAW_PREFIX = "raw__"
FIXED_GEOMETRY = "fixed_d5"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PRIMARY_MAPPING = "pooled"


class RepairError(RuntimeError):
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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(safe(dict(payload)), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def verify_source(root: Path) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    invalidation = root / "INVALIDATION.json"
    if not manifest.is_file() or not seal.is_file() or not invalidation.is_file():
        raise RepairError("source manifest, seal, or invalidation is missing")
    if sha256(manifest) != seal.read_text().split()[0]:
        raise RepairError("source manifest seal mismatch")
    payload = json.loads(manifest.read_text())
    invalid = json.loads(invalidation.read_text())
    if invalid.get("status") != "DEVELOPMENT_MAPPING_AND_SELECTION_INVALIDATED":
        raise RepairError("source invalidation contract changed")
    required = (
        "development_oof_predictions.parquet",
        "untouched_april_predictions.parquet",
    )
    for name in required:
        expected = payload.get("outputs_sha256", {}).get(name)
        if not expected or expected != sha256(root / name):
            raise RepairError(f"source output hash mismatch: {name}")
    return payload


def parse_raw_column(column: str) -> tuple[str, str, str]:
    if not column.startswith(RAW_PREFIX):
        raise ValueError(column)
    parts = column[len(RAW_PREFIX) :].split("__")
    if len(parts) != 3:
        raise ValueError(f"invalid raw prediction column: {column}")
    return str(parts[0]), str(parts[1]), str(parts[2])


def raw_column(target: str, arm: str, geometry: str) -> str:
    return RAW_PREFIX + "__".join((target, arm, geometry))


def assert_identity_order(predictions: pd.DataFrame, frame: pd.DataFrame, name: str) -> None:
    left = predictions.loc[:, list(base.IDENTITY)].reset_index(drop=True).copy()
    right = frame.loc[:, list(base.IDENTITY)].reset_index(drop=True).copy()
    for table in (left, right):
        table["__ts__"] = pd.to_datetime(table["__ts__"], utc=True)
    if not left.equals(right):
        raise RepairError(f"{name} identity/order differs from canonical panel")
    if predictions.duplicated(list(base.IDENTITY)).any():
        raise RepairError(f"{name} identities are not unique")
    if not np.allclose(
        predictions.execution_net_ev_12h.to_numpy(float),
        frame.execution_net_ev_12h.to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise RepairError(f"{name} exact-net parity failed")


def expected_tail(
    frame: pd.DataFrame,
    score: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise RepairError("tail score is non-finite or misaligned")
    rows = max(1, int(np.ceil(float(fraction) * len(frame))))
    ordered = np.sort(values)[::-1]
    cutoff = float(ordered[rows - 1])
    above = values > cutoff
    tied = np.isclose(values, cutoff, atol=1e-14, rtol=0)
    needed = rows - int(above.sum())
    if needed < 0 or needed > int(tied.sum()):
        raise RepairError("cutoff tie accounting failed")

    def expectation(column: str) -> float:
        outcome = frame[column].to_numpy(float)
        tie_mean = float(outcome[tied].mean()) if tied.any() else 0.0
        return float((outcome[above].sum() + needed * tie_mean) / rows)

    selected = base.stable_global_top_mask(frame, values, fraction)
    return {
        "fraction": float(fraction),
        "rows": rows,
        "cutoff": cutoff,
        "cutoff_tie_rows": int(tied.sum()),
        "cutoff_tie_fraction_of_book": float(tied.sum() / rows),
        "random_tie_expected_net_bps": expectation("execution_net_ev_12h")
        * 10_000.0,
        "random_tie_expected_hard0_precision": expectation(
            "opportunity_gross_above_cost_0bps"
        ),
        "random_tie_expected_hard25_precision": expectation(
            "opportunity_gross_above_cost_25bps"
        ),
        "deterministic_net_bps": float(
            frame.loc[selected, "execution_net_ev_12h"].mean() * 10_000.0
        ),
        "deterministic_long_share": float(
            frame.loc[selected, "side_name"].eq("long").mean()
        ),
    }


def raw_selection(
    predictions: pd.DataFrame,
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for column in predictions.columns:
        if not column.startswith(RAW_PREFIX):
            continue
        target, arm, geometry = parse_raw_column(column)
        for fraction in FRACTIONS:
            records.append(
                {
                    "target": target,
                    "arm": arm,
                    "geometry": geometry,
                    **expected_tail(
                        development,
                        predictions[column].to_numpy(float),
                        fraction,
                    ),
                }
            )
    metrics = pd.DataFrame(records)
    top10 = metrics.loc[np.isclose(metrics.fraction, 0.10)].copy()
    selected_arms: dict[str, list[str]] = {}
    for target in base.TARGETS:
        candidates = top10.loc[
            top10.target.eq(target)
            & top10.arm.isin(base.PRIMARY_ARMS)
            & top10.geometry.eq(FIXED_GEOMETRY)
        ].sort_values(
            [
                "random_tie_expected_net_bps",
                "random_tie_expected_hard0_precision",
                "arm",
            ],
            ascending=[False, False, True],
            kind="stable",
        )
        if len(candidates) != len(base.PRIMARY_ARMS):
            raise RepairError(f"incomplete fixed raw OOF grid for {target}")
        selected_arms[target] = candidates.arm.head(2).astype(str).tolist()

    winners: list[dict[str, Any]] = []
    for target, arms in selected_arms.items():
        for arm in arms:
            candidates = top10.loc[
                top10.target.eq(target) & top10.arm.eq(arm)
            ].sort_values(
                [
                    "random_tie_expected_net_bps",
                    "random_tie_expected_hard0_precision",
                    "geometry",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
            winners.append(candidates.iloc[0].to_dict())
    winner_frame = pd.DataFrame(winners).sort_values(
        ["target", "arm"], kind="stable"
    ).reset_index(drop=True)
    if len(winner_frame) != 2 * len(base.TARGETS):
        raise RepairError("raw OOF selection did not yield two configs per target")
    return metrics, selected_arms, winner_frame


def mapper_frame(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    output = frame.loc[
        :,
        [
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            "__decision_ts__",
            "execution_label_end_utc",
            "execution_net_ev_12h",
            "execution_gross_ev_12h",
            "execution_cost_return",
        ],
    ].copy()
    output["execution_decision_utc"] = pd.to_datetime(
        output.pop("__decision_ts__"), utc=True
    )
    output["raw_score"] = np.asarray(score, dtype=float)
    output["score_available_utc"] = pd.to_datetime(
        output.execution_decision_utc, utc=True
    )
    return output


def causal_forward_maps(
    development: pd.DataFrame,
    april: pd.DataFrame,
    development_score: np.ndarray,
    april_score: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = mapper_frame(development, development_score)
    evaluate = mapper_frame(april, april_score)
    history = pd.concat([reference, evaluate], ignore_index=True, sort=False)
    pooled, pooled_audit = mapping.causal_map(
        history, evaluate, add_side_residual=False
    )
    side, side_audit = mapping.causal_map(
        history, evaluate, add_side_residual=True
    )
    added = [
        "causal_pooled_side_21d",
        "causal_pooled_side_21d_eligible",
        "causal_pooled_side_21d_status",
        "causal_pooled_side_21d_pooled_rows",
        "causal_pooled_side_21d_side_rows",
        "causal_pooled_side_21d_side_weight",
        "causal_pooled_side_21d_snapshot_utc",
    ]
    result = pooled.merge(
        side.loc[:, ["candidate_id", "side_name", *added]],
        on=["candidate_id", "side_name"],
        how="left",
        validate="one_to_one",
    )
    if not pooled_audit.strict_causal_window_pass.all():
        raise RepairError("pooled causal map violates the time window")
    if not side_audit.strict_causal_window_pass.all():
        raise RepairError("side-residual causal map violates the time window")
    return result, pd.concat(
        [
            pooled_audit.assign(mapping_kind="pooled"),
            side_audit.assign(mapping_kind="pooled_plus_shrunk_side"),
        ],
        ignore_index=True,
    )


def evaluate_maps(
    frame: pd.DataFrame,
    config: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for kind, score, eligible in (
        ("raw", "raw_score", None),
        ("pooled", "causal_pooled_21d", "causal_pooled_21d_eligible"),
        (
            "pooled_plus_shrunk_side",
            "causal_pooled_side_21d",
            "causal_pooled_side_21d_eligible",
        ),
    ):
        result = mapping.evaluate_arm(
            frame,
            arm=f"{config}__{kind}",
            score=score,
            eligible=eligible,
        )
        for table in result:
            for row in table:
                row.update(config=config, mapping_kind=kind)
        metrics.extend(result[0])
        sides.extend(result[1])
        assets.extend(result[2])
        intervals.extend(result[3])
    return metrics, sides, assets, intervals


def promotion_gates(
    metrics: pd.DataFrame,
    sides: pd.DataFrame,
    controls: pd.DataFrame,
    configs: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    control = controls.loc[
        controls.control.eq("base")
        & controls.mapping_kind.eq(PRIMARY_MAPPING)
        & np.isclose(controls.top_fraction, 0.10)
    ].iloc[0]
    for config in configs:
        metric = metrics.loc[
            metrics.config.eq(config)
            & metrics.mapping_kind.eq(PRIMARY_MAPPING)
            & np.isclose(metrics.top_fraction, 0.10)
        ].iloc[0]
        side = sides.loc[
            sides.config.eq(config) & sides.mapping_kind.eq(PRIMARY_MAPPING)
        ]
        checks = {
            "april_is_new_untouched": False,
            "top10_positive": bool(metric.random_tie_expected_net_bps > 0),
            "latest_week_positive": bool(metric.latest_week_net_bps > 0),
            "cutoff_tie_fraction_le_5pct": bool(
                metric.cutoff_tie_fraction_of_book <= 0.05
            ),
            "largest_side_share_le_75pct": bool(
                len(side) == 2 and side.share.max() <= 0.75
            ),
            "both_sides_positive": bool(
                len(side) == 2 and side.net_bps.gt(0).all()
            ),
            "beats_mapped_base_control": bool(
                metric.random_tie_expected_net_bps
                > control.random_tie_expected_net_bps
            ),
        }
        records.append(
            {
                "config": config,
                "mapping_kind": PRIMARY_MAPPING,
                "top10_net_bps": float(metric.random_tie_expected_net_bps),
                "latest_week_net_bps": float(metric.latest_week_net_bps),
                "mapped_base_control_bps": float(
                    control.random_tie_expected_net_bps
                ),
                "checks_json": json.dumps(checks, sort_keys=True),
                "all_model_economic_checks_pass": bool(
                    all(value for key, value in checks.items() if key != "april_is_new_untouched")
                ),
                "promotion_eligible": False,
                "portfolio_replay_authorized": False,
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_manifest = verify_source(args.source)
    frame, panel_manifest = base.load_panel(args.panel_root)
    development, april = base.split_development_april(frame)
    oof = pd.read_parquet(args.source / "development_oof_predictions.parquet")
    old_april = pd.read_parquet(args.source / "untouched_april_predictions.parquet")
    assert_identity_order(oof, development, "development OOF")
    assert_identity_order(old_april, april, "April source")
    raw_metrics, selected_arms, selected = raw_selection(oof, development)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        model_dir = stage / "models"
        model_dir.mkdir()
        geometry_by_name = {item.name: item for item in base.GEOMETRIES}
        metrics: list[dict[str, Any]] = []
        sides: list[dict[str, Any]] = []
        assets: list[dict[str, Any]] = []
        intervals: list[dict[str, Any]] = []
        predictions: list[pd.DataFrame] = []
        audits: list[pd.DataFrame] = []
        reuse: list[dict[str, Any]] = []
        configs: list[str] = []

        for index, row in enumerate(selected.itertuples(index=False)):
            target = str(row.target)
            arm = str(row.arm)
            geometry_name = str(row.geometry)
            config = "__".join((target, arm, geometry_name))
            configs.append(config)
            column = raw_column(target, arm, geometry_name)
            development_score = oof[column].to_numpy(float)
            if column in old_april:
                april_score = old_april[column].to_numpy(float)
                source_kind = "reused_hash_verified_v1_forward_prediction"
                for side_name in base.SIDES:
                    source_model = (
                        args.source
                        / "models"
                        / f"{config}__{side_name}.cbm"
                    )
                    if not source_model.is_file():
                        raise RepairError(f"reused model is missing: {source_model}")
                    shutil.copy2(
                        source_model,
                        model_dir / source_model.name,
                    )
            else:
                april_score = base.fit_final_side_models(
                    development,
                    april,
                    arm=arm,
                    target=target,
                    geometry=geometry_by_name[geometry_name],
                    threads=args.threads,
                    seed=args.seed,
                    model_dir=model_dir,
                )
                source_kind = "new_missing_forward_fit_after_raw_oof_freeze"
            mapped, audit = causal_forward_maps(
                development,
                april,
                development_score,
                april_score,
            )
            result = evaluate_maps(mapped, config)
            metrics.extend(result[0])
            sides.extend(result[1])
            assets.extend(result[2])
            intervals.extend(result[3])
            predictions.append(mapped.assign(config=config))
            audits.append(audit.assign(config=config))
            reuse.append(
                {
                    "config": config,
                    "forward_source": source_kind,
                    "development_raw_column": column,
                    "development_raw_sha256": hashlib.sha256(
                        np.asarray(development_score, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "april_raw_sha256": hashlib.sha256(
                        np.asarray(april_score, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )

        metric_frame = pd.DataFrame(metrics)
        side_frame = pd.DataFrame(sides)
        asset_frame = pd.DataFrame(assets)
        interval_frame = pd.DataFrame(intervals)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        audit_frame = pd.concat(audits, ignore_index=True)

        control_metrics: list[dict[str, Any]] = []
        for control_name, score_column in (("base", "base_oof_score"),):
            control_mapped, control_audit = causal_forward_maps(
                development,
                april,
                development[score_column].to_numpy(float),
                april[score_column].to_numpy(float),
            )
            result = evaluate_maps(control_mapped, f"control__{control_name}")
            for row in result[0]:
                row["control"] = control_name
                control_metrics.append(row)
            audits.append(control_audit.assign(config=f"control__{control_name}"))
        control_frame = pd.DataFrame(control_metrics)
        audit_frame = pd.concat(
            [audit_frame, audits[-1].assign(config="control__base")],
            ignore_index=True,
        )
        gates = promotion_gates(
            metric_frame, side_frame, control_frame, configs
        )

        raw_metrics.to_csv(stage / "development_raw_oof_selection_metrics.csv", index=False)
        selected.to_csv(stage / "raw_oof_selected_configs.csv", index=False)
        pd.DataFrame(reuse).to_csv(stage / "forward_prediction_provenance.csv", index=False)
        metric_frame.to_csv(stage / "april_global_metrics.csv", index=False)
        side_frame.to_csv(stage / "april_side_top10.csv", index=False)
        asset_frame.to_csv(stage / "april_asset_top10.csv", index=False)
        interval_frame.to_csv(stage / "april_day_block_intervals.csv", index=False)
        control_frame.to_csv(stage / "april_control_metrics.csv", index=False)
        gates.to_csv(stage / "promotion_gates.csv", index=False)
        prediction_frame.to_parquet(
            stage / "april_predictions.parquet", index=False, compression="zstd"
        )
        audit_frame.to_parquet(
            stage / "mapping_audit.parquet", index=False, compression="zstd"
        )
        outputs = {
            str(path.relative_to(stage)): sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": SCHEMA,
            "run_id": args.output_dir.name,
            "status": "COMPLETED_RAW_OOF_SELECTION_REPAIR_REUSED_APRIL_NO_PROMOTION",
            "promotion_eligible": False,
            "portfolio_replay": "NOT_RUN",
            "repair": {
                "source_v1_status": "mapped development selection invalidated",
                "preserved_source_evidence": "raw side-local OOF predictions and predeclared forward fits",
                "selection": "raw OOF random-tie-expected pooled-global top10 only; no mapped development economics",
                "feature_top_two_by_target": selected_arms,
                "selected_configs": selected.to_dict(orient="records"),
                "mapping_after_freeze": (
                    "score-specific causal daily 21d pooled isotonic primary; "
                    "shrunk-side residual reported separately"
                ),
                "april_evidence": "reused diagnostic only",
            },
            "population": {
                "development_rows": len(development),
                "april_rows": len(april),
                "development_side_rows": development.groupby("side_name").size().to_dict(),
                "april_side_rows": april.groupby("side_name").size().to_dict(),
            },
            "validation": {
                "raw_oof_source": "five contiguous complement blocks with exact 12h two-sided path purge",
                "not_walk_forward_evidence": True,
                "mapper_selection_separation": "all target/arm/geometry choices frozen before any causal mapping",
                "april_is_new_untouched": False,
            },
            "selection_contract": {
                "scope": "one pooled global book across all timestamps and both sides",
                "fractions": list(FRACTIONS),
                "tie_metric": "random tie expectation; candidate_id deterministic ledger retained",
                "never_per_timestamp_or_side": True,
            },
            "input_sha256": {
                "source_manifest": sha256(args.source / "manifest.json"),
                "source_invalidation": sha256(args.source / "INVALIDATION.json"),
                "source_development_oof": sha256(
                    args.source / "development_oof_predictions.parquet"
                ),
                "source_april_predictions": sha256(
                    args.source / "untouched_april_predictions.parquet"
                ),
                "panel_manifest": sha256(args.panel_root / "manifest.json"),
                "panel": panel_manifest["outputs_sha256"]["panel.parquet"],
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "April has already been inspected and cannot promote any configuration.",
                "The repaired support heads are a precursor to, not a substitute for, clean/competing-risk/capture reliability.",
                "No proper 2025 pre-exit capture or severe-loss OOF head is introduced here.",
                "Timing, MAE, target-price and wait actions remain outside this EV layer.",
            ],
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
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--panel-root", type=Path, default=PANEL)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--seed", type=int, default=base.SEED)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
