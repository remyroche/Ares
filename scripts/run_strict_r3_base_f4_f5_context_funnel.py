#!/usr/bin/env python3
"""Offline F4/F5 causal-context base screen for long Strict-R3.

F4 appends causal execution-readiness transforms of book, basis and flow
state. F5 appends causal asset-versus-market divergence transforms.  These are
base-only screening arms: no residual, MC1, admission, execution or live
artifact is touched.  A successful arm must still pass the predeclared
downstream base-and-residual propagation gate before any promotion.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features import (  # noqa: E402
    STRICT_R3_F4_EXECUTION_CONTEXT_SOURCE_KEYS,
    STRICT_R3_F5_ASSET_DIVERGENCE_SOURCE_KEYS,
    strict_r3_execution_divergence_features,
)
from extreme_price_movements.strict_r3_canonical_v2 import _fit_medians, _numeric_matrix  # noqa: E402
from scripts.run_strict_r3_base_f1_session_funnel import (  # noqa: E402
    MAX_TRAIN_ROWS,
    _base_params,
    _d2_weights,
    _diagnose,
    _feature_contract,
    _strict_train,
)
from scripts.run_strict_r3_base_f2_f3_context_funnel import _gate  # noqa: E402
from scripts.run_strict_r3_base_recall_funnel import (  # noqa: E402
    BASE_ROUTE_FRACTION,
    DEFAULT_CONTROL,
    DEFAULT_SOURCE,
    PERIODS,
    _utc,
    timestamp_route,
)


DEFAULT_B0_ROOT = ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1"
ROLLING_CONTEXT_HOURS = 24 * 21
F4_FIELDS = tuple(
    f"f4_{key}_{suffix}"
    for key in STRICT_R3_F4_EXECUTION_CONTEXT_SOURCE_KEYS
    for suffix in ("delta_1h", "delta_4h", "minus_mean_7d", "cdf_7d")
)
F5_FIELDS = tuple(
    f"f5_{key}_{suffix}"
    for key in STRICT_R3_F5_ASSET_DIVERGENCE_SOURCE_KEYS
    for suffix in ("cross_section_rank", "minus_cross_section_median", "delta_4h", "minus_mean_7d")
)


def _require_exact_hourly_phase_zero(frame: pd.DataFrame, *, source_name: str) -> None:
    """F4/F5 are valid only for the declared UTC :00 research population."""

    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    nonzero = (timestamp.dt.minute != 0) | (timestamp.dt.second != 0) | (timestamp.dt.microsecond != 0)
    if nonzero.any():
        sample = timestamp.loc[nonzero].head(5).astype(str).tolist()
        raise AssertionError(f"{source_name} contains non-:00 decision timestamps: {sample}")


def _source_columns(
    fields: tuple[str, ...],
    *,
    context_source_keys: tuple[str, ...],
    include_training_labels: bool,
) -> list[str]:
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", *fields,
        *context_source_keys,
    ]
    if include_training_labels:
        columns.extend(["r3_class", "r3_label_available_ts", "prequential_base_rank42"])
    return list(dict.fromkeys(columns))


def _load_source(
    source: Path,
    fields: tuple[str, ...],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    context_source_keys: tuple[str, ...],
    include_f4: bool,
    include_f5: bool,
    include_training_labels: bool,
) -> pd.DataFrame:
    """Materialise one causal window plus rolling context, never full history.

    The F4/F5 transforms need only their preceding own-asset records and the
    complete contemporaneous cross-section.  A 21-day causal prefix safely
    exceeds the 168-record rolling primitive on the hourly candidate panel,
    while avoiding the former 3m-row/184-column resident frame.
    """

    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("__decision_ts__", ">=", _utc(start)))
    if end is not None:
        filters.append(("__decision_ts__", "<", _utc(end)))
    frame = pd.read_parquet(
        source,
        columns=_source_columns(
            fields,
            context_source_keys=context_source_keys,
            include_training_labels=include_training_labels,
        ),
        filters=filters or None,
    )
    if frame.empty:
        raise AssertionError("F4/F5 causal window is empty")
    columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__",
    ]))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if include_training_labels:
        frame["r3_label_available_ts"] = pd.to_datetime(
            frame["r3_label_available_ts"], utc=True, errors="coerce",
        )
    _require_exact_hourly_phase_zero(frame, source_name="F4/F5 source")
    derived = strict_r3_execution_divergence_features(
        frame, include_f4=include_f4, include_f5=include_f5,
    )
    for name, values in derived.items():
        frame[name] = values
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("source candidate IDs must be unique")
    return frame


def _load_training_index(source: Path) -> pd.DataFrame:
    """Load only identity/label columns needed to select each capped fold."""

    frame = pd.read_parquet(
        source,
        columns=[
            "candidate_id", "__decision_ts__", "r3_class",
            "r3_label_available_ts",
        ],
    )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["r3_label_available_ts"] = pd.to_datetime(
        frame["r3_label_available_ts"], utc=True, errors="coerce",
    )
    _require_exact_hourly_phase_zero(frame, source_name="F4/F5 training index")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("F4/F5 source candidate IDs must be globally unique")
    return frame


def _training_ids(index: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    reserve_start = _utc(cutoff) - pd.Timedelta(days=28)
    selected = index.loc[
        index["__decision_ts__"].lt(reserve_start)
        & index["r3_label_available_ts"].lt(reserve_start)
        & index["r3_class"].isin([0, 1, 2]),
    ].sort_values("r3_label_available_ts", kind="stable").tail(MAX_TRAIN_ROWS).copy()
    if selected.empty:
        raise AssertionError(f"no resolved F4/F5 training rows before {cutoff.isoformat()}")
    return selected.reset_index(drop=True)


def _select_ids(window: pd.DataFrame, candidate_ids: pd.Series, *, label: str) -> pd.DataFrame:
    indexed = window.set_index("candidate_id", drop=False)
    expected = pd.Index(candidate_ids.astype(str))
    missing = expected.difference(indexed.index)
    if len(missing):
        raise AssertionError(f"{label} window misses {len(missing)} required frozen candidate IDs")
    selected = indexed.loc[expected].copy().reset_index(drop=True)
    if selected["candidate_id"].duplicated().any() or len(selected) != len(expected):
        raise AssertionError(f"{label} selection changed candidate identities")
    return selected


def _coverage_rows(
    frame: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    block: str,
    scope: str,
    period: str,
) -> list[dict[str, object]]:
    return [
        {
            "block": block,
            "scope": scope,
            "period": period,
            "feature": field,
            "non_null_rows": int(frame[field].notna().sum()),
            "rows": int(len(frame)),
        }
        for field in fields
    ]


def _derive_window(
    source: Path,
    base_fields: tuple[str, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    arm: str,
    include_training_labels: bool,
) -> pd.DataFrame:
    """Read and derive one bounded causal panel.

    This intentionally performs the rolling/cross-sectional transforms before
    selecting either training or held candidate IDs.  The 21-day prefix gives
    the causal rolling primitives their required history while preserving the
    full contemporaneous asset universe for each timestamp.
    """

    if arm == "F4_execution_context":
        context_keys = STRICT_R3_F4_EXECUTION_CONTEXT_SOURCE_KEYS
        expected_fields = F4_FIELDS
        include_f4, include_f5 = True, False
    elif arm == "F5_asset_divergence":
        context_keys = STRICT_R3_F5_ASSET_DIVERGENCE_SOURCE_KEYS
        expected_fields = F5_FIELDS
        include_f4, include_f5 = False, True
    else:
        raise ValueError(f"unknown F4/F5 arm: {arm}")
    window = _load_source(
        source,
        base_fields,
        start=start,
        end=end,
        context_source_keys=context_keys,
        include_f4=include_f4,
        include_f5=include_f5,
        include_training_labels=include_training_labels,
    )
    missing = sorted(set(expected_fields).difference(window.columns))
    if missing:
        raise AssertionError(f"F4/F5 derivation misses fields: {missing[:5]}")
    # Held rows must remain target-free until the post-score diagnostics join.
    if not include_training_labels and {"r3_class", "r3_label_available_ts"}.intersection(window.columns):
        raise AssertionError("held F4/F5 window illegally contains training labels")
    return window


def _checkpoint_paths(out_dir: Path, block: str) -> tuple[Path, Path]:
    root = out_dir / "checkpoints"
    return root / f"{block}_scores.parquet", root / f"{block}_audit.json"


def _write_checkpoint(out_dir: Path, block: str, scores: pd.DataFrame, audit: dict[str, object]) -> None:
    score_path, audit_path = _checkpoint_paths(out_dir, block)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(score_path, index=False, compression="zstd")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _read_checkpoint(out_dir: Path, block: str) -> tuple[pd.DataFrame, dict[str, object]] | None:
    score_path, audit_path = _checkpoint_paths(out_dir, block)
    if not score_path.is_file() and not audit_path.is_file():
        return None
    if not score_path.is_file() or not audit_path.is_file():
        raise AssertionError(f"incomplete F4/F5 checkpoint for {block}")
    return pd.read_parquet(score_path), json.loads(audit_path.read_text())


def _seal_existing_run(
    *,
    out_dir: Path,
    source: Path,
    b0_root: Path,
    base_fields: tuple[str, ...],
) -> None:
    """Seal an interrupted finalisation without rewriting frozen score tables."""

    required = {
        "scores": out_dir / "f4_f5_target_free_scores.parquet",
        "scored": out_dir / "f4_f5_outcome_joined_audit.parquet",
        "audit": out_dir / "f4_f5_block_training_audit.parquet",
        "coverage": out_dir / "f4_f5_feature_coverage_by_period.parquet",
        "metrics": out_dir / "f4_f5_base_metrics.parquet",
        "gates": out_dir / "f4_f5_advancement_gates.parquet",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot seal incomplete F4/F5 run: {missing}")
    scores = pd.read_parquet(required["scores"])
    scored = pd.read_parquet(required["scored"], columns=["candidate_id", "__decision_ts__"])
    timestamps = pd.to_datetime(scores["__decision_ts__"], utc=True, errors="raise")
    _require_exact_hourly_phase_zero(scores, source_name="F4/F5 final target-free scores")
    if scores["candidate_id"].duplicated().any() or len(scores) != len(scored):
        raise AssertionError("F4/F5 final score identity is not one-to-one with outcome audit")
    if set(scores["candidate_id"].astype(str)) != set(scored["candidate_id"].astype(str)):
        raise AssertionError("F4/F5 outcome audit changed target-free candidate identity")
    audit = pd.read_parquet(required["audit"])
    if len(audit) != 21 or not audit["all_labels_before_reserve"].fillna(False).all():
        raise AssertionError("F4/F5 final audit lacks 21 strict-reserve blocks")
    rejected_arms: dict[str, dict[str, object]] = {}
    for path in sorted((out_dir / "checkpoints").glob("*_audit.json")):
        rejected_arms.update(json.loads(path.read_text()).get("arm_rejections", {}))
    if "F5_asset_divergence" not in rejected_arms:
        raise AssertionError("seal-only expects the recorded F5 coverage rejection")
    coverage = pd.read_parquet(required["coverage"])
    f4_coverage = coverage.loc[
        coverage["scope"].eq("held") & coverage["feature"].isin(F4_FIELDS), "coverage",
    ]
    if f4_coverage.empty or f4_coverage.lt(.90).any():
        raise AssertionError("F4 fails the held coverage gate during sealing")
    metrics = pd.read_parquet(required["metrics"])
    gates = pd.read_parquet(required["gates"])
    active_arms = {"F4_execution_context": F4_FIELDS}
    quarterly = metrics.loc[
        metrics["label"].str.match(r"^20\d\dQ[1-4]$", na=False)
    ].pivot(index="label", columns="arm", values="recall_policy_ge_100")
    f4_gate = gates.loc[gates["arm"].eq("F4_execution_context")]
    f4_advance = bool(
        f4_gate["relative_recall_gain"].ge(.02).all()
        and f4_gate["mean_policy_net_delta_bps"].ge(-5.0).all()
        and f4_gate["rank_ic_delta"].ge(-.005).all()
        and (quarterly["F4_execution_context"] >= quarterly["B0"]).all()
    )
    manifest = {
        "schema": "strict_r3_long_f4_f5_context_v2_canonical_d2",
        "scope": "offline base-only feature-block screening; no residual, MC1, portfolio or live artifact modified",
        "source": str(source), "b0_root": str(b0_root),
        "base_feature_count": len(base_fields), "f4_feature_count": len(F4_FIELDS), "f5_feature_count": len(F5_FIELDS),
        "causality": "F4/F5 inputs are target-free transforms of frozen decision-time primitives; fitting requires R3 label availability before the 28-day calibration reserve and uses the canonical D2 teacher weighting",
        "feature_coverage_gate": ">=90% for every train/held fold and reported period",
        "rejected_feature_arms": rejected_arms,
        "base_training_contract": "same 240k latest-label-resolved cap, train-fold median imputation, D2 teacher weighting, and fully excluded 28-day reserve as B0",
        "materialisation": "per-block target-free held panels and label-resolved training panels; each carries a 21-day causal prefix before fully-universe rolling/cross-sectional F4/F5 derivation",
        "checkpointing": "immutable completed per-block score/audit checkpoints; resume validates candidate identity before reuse",
        "advance_to_downstream_rebuild": {"F4_execution_context": f4_advance, "F5_asset_divergence": False},
        "next_required_stage": "For any advancing block: compare base_only against base_and_residual using a full strict-prequential downstream rebuild; do not inject raw F4/F5 fields into MC1 first.",
        "seal_only": True,
        "score_rows": int(len(scores)),
        "score_time_min": timestamps.min().isoformat(),
        "score_time_max": timestamps.max().isoformat(),
    }
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("F4/F5 run manifest already exists")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "sealed", "rows": int(len(scores)), "advance": manifest["advance_to_downstream_rebuild"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume only immutable completed per-block checkpoints from this exact run",
    )
    parser.add_argument(
        "--seal-only", action="store_true",
        help="validate existing final tables and write only a missing run manifest",
    )
    args = parser.parse_args()
    run_state_path = args.out_dir / "run_state.json"
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if args.out_dir.exists() and args.resume and not run_state_path.is_file():
        raise AssertionError("F4/F5 resume requires a prior run_state.json")
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)
        run_state_path.write_text(json.dumps({
            "schema": "strict_r3_long_f4_f5_context_v2_canonical_d2",
            "source": str(args.source),
            "control_root": str(args.control_root),
            "b0_root": str(args.b0_root),
            "causal_prefix_hours": ROLLING_CONTEXT_HOURS,
        }, indent=2, sort_keys=True) + "\n")
    else:
        state = json.loads(run_state_path.read_text())
        expected = {"source": str(args.source), "control_root": str(args.control_root), "b0_root": str(args.b0_root)}
        if any(state.get(key) != value for key, value in expected.items()):
            raise AssertionError("F4/F5 resume paths do not match the sealed run state")

    base_fields = _feature_contract(args.control_root)
    if args.seal_only:
        if not args.resume:
            raise AssertionError("F4/F5 seal-only requires --resume for an existing immutable run")
        _seal_existing_run(
            out_dir=args.out_dir, source=args.source, b0_root=args.b0_root,
            base_fields=base_fields,
        )
        return
    if len(F4_FIELDS) != 32 or len(F5_FIELDS) != 40:
        raise AssertionError(f"unexpected declared F4/F5 field counts: {len(F4_FIELDS)}, {len(F5_FIELDS)}")
    arms = {"F4_execution_context": F4_FIELDS, "F5_asset_divergence": F5_FIELDS}

    b0 = pd.read_parquet(
        args.b0_root / "b0_target_free_reconstruction.parquet",
        columns=["candidate_id", "__decision_ts__", "control_block"],
    )
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    _require_exact_hourly_phase_zero(b0, source_name="frozen B0 control")
    if b0["candidate_id"].duplicated().any():
        raise AssertionError("frozen B0 candidate identities must be unique")
    training_index = _load_training_index(args.source)

    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    rejected_arms: dict[str, dict[str, object]] = {}
    for path in sorted(args.control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), ["candidate_id", "__decision_ts__"]].copy()
        if held_ids.empty:
            continue
        checkpoint = _read_checkpoint(args.out_dir, block) if args.resume else None
        if checkpoint is not None:
            output, audit = checkpoint
            expected_ids = pd.Index(held_ids["candidate_id"].astype(str))
            if set(output["candidate_id"].astype(str)) != set(expected_ids) or len(output) != len(expected_ids):
                raise AssertionError(f"checkpoint identity mismatch for {block}")
            rows.append(output)
            audits.append(audit)
            rejected_arms.update(audit.get("arm_rejections", {}))
            continue

        reserve_start = cutoff - pd.Timedelta(days=28)
        train_ids = _training_ids(training_index, cutoff)
        train_start = _utc(train_ids["__decision_ts__"].min()) - pd.Timedelta(hours=ROLLING_CONTEXT_HOURS)
        held_start = _utc(held_ids["__decision_ts__"].min()) - pd.Timedelta(hours=ROLLING_CONTEXT_HOURS)
        held_end = _utc(held_ids["__decision_ts__"].max()) + pd.Timedelta(hours=1)
        output: pd.DataFrame | None = None
        audit: dict[str, object] = {
            "block": block,
            "cutoff": cutoff.isoformat(),
            "held_rows": int(len(held_ids)),
            "train_rows": int(len(train_ids)),
            "reserve_start": reserve_start.isoformat(),
            "all_labels_before_reserve": True,
            "train_context_start": train_start.isoformat(),
            "held_context_start": held_start.isoformat(),
            "held_context_end_exclusive": held_end.isoformat(),
            "coverage_rows": [],
            "arm_rejections": {},
        }
        for arm, extras in arms.items():
            if arm in rejected_arms:
                continue
            # F4 and F5 are deliberately materialised separately.  This keeps
            # each bounded panel independent and avoids retaining the other
            # arm's 40/32 derived fields while LightGBM is fitting.
            train_window = _derive_window(
                args.source, base_fields, start=train_start, end=reserve_start,
                arm=arm, include_training_labels=True,
            )
            train = _select_ids(train_window, train_ids["candidate_id"], label=f"{block} {arm} training")
            # The full-universe prefix is required only to derive the causal
            # context.  Once frozen training IDs have been selected it must
            # not remain resident while the held panel is built or LGBM fits.
            del train_window
            gc.collect()
            if not train["r3_label_available_ts"].lt(reserve_start).all():
                raise AssertionError(f"{block} {arm} training labels enter the 28-day reserve")
            if not train["r3_class"].isin([0, 1, 2]).all():
                raise AssertionError(f"{block} {arm} training labels are not canonical R3 classes")
            held_window = _derive_window(
                args.source, base_fields, start=held_start, end=held_end,
                arm=arm, include_training_labels=False,
            )
            held = _select_ids(held_window, held_ids["candidate_id"], label=f"{block} {arm} held")
            del held_window
            gc.collect()
            sample_weight, weight_audit = _d2_weights(train)
            train_coverage = float(train.loc[:, extras].notna().mean().min())
            held_coverage = float(held.loc[:, extras].notna().mean().min())
            audit["coverage_rows"].extend(_coverage_rows(
                train, extras, block=block, scope="train", period=block,
            ))
            for period, (start, end) in PERIODS.items():
                period_held = held.loc[
                    held["__decision_ts__"].ge(_utc(start)) & held["__decision_ts__"].lt(_utc(end))
                ]
                if not period_held.empty:
                    audit["coverage_rows"].extend(_coverage_rows(
                        period_held, extras, block=block, scope="held", period=period,
                    ))
            if min(train_coverage, held_coverage) < .90:
                rejection = {
                    "reason": "per_fold_feature_coverage_below_90pct",
                    "block": block,
                    "train_coverage": train_coverage,
                    "held_coverage": held_coverage,
                }
                # A failed optional feature arm is an ablation result, not a
                # reason to discard an otherwise independent arm.  The
                # rejection is persisted and that arm is never scored again.
                rejected_arms[arm] = rejection
                audit["arm_rejections"][arm] = rejection
                audit[f"{arm}_min_train_feature_coverage"] = train_coverage
                audit[f"{arm}_min_held_feature_coverage"] = held_coverage
                del train, held, sample_weight
                gc.collect()
                continue
            contract = (*base_fields, *extras)
            medians = _fit_medians(train, contract)
            model = lgb.LGBMClassifier(**_base_params(bundle)).fit(
                _numeric_matrix(train, contract, medians),
                train["r3_class"].astype(int),
                sample_weight=sample_weight,
            )
            proba = model.predict_proba(_numeric_matrix(held, contract, medians))
            arm_identity = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
            if output is None:
                output = arm_identity
            elif not output.loc[:, ["candidate_id", "__decision_ts__"]].equals(arm_identity):
                raise AssertionError(f"{block} F4/F5 arm materialisation changed held identity/order")
            output[f"{arm}_p_adverse"] = proba[:, 0]
            output[f"{arm}_p_weak"] = proba[:, 1]
            output[f"{arm}_p_clear"] = proba[:, 2]
            output[f"{arm}_score"] = proba[:, 2] - .5 * proba[:, 0]
            audit[f"{arm}_min_train_feature_coverage"] = train_coverage
            audit[f"{arm}_min_held_feature_coverage"] = held_coverage
            audit[f"{arm}_d2_weight_audit_json"] = json.dumps(weight_audit, sort_keys=True)
            del train, held, model, proba, medians, sample_weight
            gc.collect()
        if output is None:
            raise AssertionError(f"{block} has no F4/F5 output")
        output["control_block"] = block
        _write_checkpoint(args.out_dir, block, output, audit)
        rows.append(output)
        audits.append(audit)
        del output
        gc.collect()

    predictions = pd.concat(rows, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if len(predictions) != len(b0) or predictions["candidate_id"].duplicated().any():
        raise AssertionError("F4/F5 predictions must preserve frozen B0 candidate identities")
    outcome = pd.read_parquet(
        args.b0_root / "outcome_joined_recall_ledger.parquet",
        columns=[
            "candidate_id", "__decision_ts__", "control_block", "base_score",
            "policy_path_valid", "policy_net_bps", "policy_ge_50", "policy_ge_100",
            "policy_ge_200", "positive_top20", "positive_top10",
        ],
    )
    outcome["__decision_ts__"] = pd.to_datetime(outcome["__decision_ts__"], utc=True, errors="raise")
    scored = outcome.merge(
        predictions, on=["candidate_id", "__decision_ts__", "control_block"], how="inner", validate="one_to_one",
    )
    if len(scored) != len(b0):
        raise AssertionError("F4/F5 outcome join must preserve frozen B0 identities")
    active_arms = {arm: extras for arm, extras in arms.items() if arm not in rejected_arms}
    scored["B0_route"] = timestamp_route(scored, "base_score", fraction=BASE_ROUTE_FRACTION)
    for arm in active_arms:
        scored[f"{arm}_route"] = timestamp_route(scored, f"{arm}_score", fraction=BASE_ROUTE_FRACTION)

    metric_rows: list[dict[str, object]] = []
    labels: list[tuple[str, pd.DataFrame]] = []
    labels.extend(
        (
            name,
            scored.loc[
                scored["__decision_ts__"].ge(_utc(start))
                & scored["__decision_ts__"].lt(_utc(end))
            ].copy(),
        )
        for name, (start, end) in PERIODS.items()
    )
    labels.extend(
        (str(q), group.copy())
        for q, group in scored.groupby(scored["__decision_ts__"].dt.to_period("Q"), sort=True)
        if q >= pd.Period("2025Q4", freq="Q")
    )
    for label, subset in labels:
        for arm, score, route in [
            ("B0", "base_score", "B0_route"),
            *[(name, f"{name}_score", f"{name}_route") for name in active_arms],
        ]:
            row = _diagnose(subset, subset[route].to_numpy(bool), score, label)
            row["arm"] = arm
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    gates = pd.concat([_gate(metrics, arm) for arm in active_arms], ignore_index=True) if active_arms else pd.DataFrame()
    quarterly = metrics.loc[
        metrics["label"].str.match(r"^20\d\dQ[1-4]$", na=False)
    ].pivot(index="label", columns="arm", values="recall_policy_ge_100")
    decisions: dict[str, bool] = {}
    for arm in active_arms:
        arm_gate = gates.loc[gates["arm"].eq(arm)]
        decisions[arm] = bool(
            arm_gate["relative_recall_gain"].ge(.02).all()
            and arm_gate["mean_policy_net_delta_bps"].ge(-5.0).all()
            and arm_gate["rank_ic_delta"].ge(-.005).all()
            and (quarterly[arm] >= quarterly["B0"]).all()
        )

    coverage_rows = [row for audit in audits for row in audit["coverage_rows"]]
    coverage_by_period = pd.DataFrame(coverage_rows)
    coverage_by_period = coverage_by_period.groupby(
        ["scope", "period", "feature"], as_index=False,
    )[["non_null_rows", "rows"]].sum()
    coverage_by_period["coverage"] = coverage_by_period["non_null_rows"] / coverage_by_period["rows"]
    held_coverage = coverage_by_period.loc[coverage_by_period["scope"].eq("held")]
    for arm, extras in active_arms.items():
        failed = held_coverage.loc[
            held_coverage["feature"].isin(extras) & held_coverage["coverage"].lt(.90)
        ]
        if not failed.empty:
            raise AssertionError(f"{arm} fails >=90% held-period coverage: {failed.to_dict('records')}")
    for arm in rejected_arms:
        decisions[arm] = False
    predictions.to_parquet(args.out_dir / "f4_f5_target_free_scores.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out_dir / "f4_f5_outcome_joined_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "f4_f5_block_training_audit.parquet", index=False)
    coverage_by_period.to_parquet(args.out_dir / "f4_f5_feature_coverage_by_period.parquet", index=False)
    metrics.to_parquet(args.out_dir / "f4_f5_base_metrics.parquet", index=False)
    gates.to_parquet(args.out_dir / "f4_f5_advancement_gates.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_f4_f5_context_v2_canonical_d2",
        "scope": "offline base-only feature-block screening; no residual, MC1, portfolio or live artifact modified",
        "source": str(args.source), "b0_root": str(args.b0_root),
        "base_feature_count": len(base_fields), "f4_feature_count": len(F4_FIELDS), "f5_feature_count": len(F5_FIELDS),
        "causality": "F4/F5 inputs are target-free transforms of frozen decision-time primitives; fitting requires R3 label availability before the 28-day calibration reserve and uses the canonical D2 teacher weighting",
        "feature_coverage_gate": ">=90% for every train/held fold and reported period",
        "rejected_feature_arms": rejected_arms,
        "base_training_contract": "same 240k latest-label-resolved cap, train-fold median imputation, D2 teacher weighting, and fully excluded 28-day reserve as B0",
        "materialisation": "per-block target-free held panels and label-resolved training panels; each carries a 21-day causal prefix before fully-universe rolling/cross-sectional F4/F5 derivation",
        "checkpointing": "immutable completed per-block score/audit checkpoints; resume validates candidate identity before reuse",
        "advance_to_downstream_rebuild": decisions,
        "next_required_stage": "For any advancing block: compare base_only against base_and_residual using a full strict-prequential downstream rebuild; do not inject raw F4/F5 fields into MC1 first.",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(scored)), "advance": decisions}, sort_keys=True))


if __name__ == "__main__":
    main()
