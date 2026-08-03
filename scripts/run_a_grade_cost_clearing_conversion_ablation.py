#!/usr/bin/env python3
"""A-grade hourly IC→EV conversion ablation with causal context arms.

The baseline and every challenger use exactly the same context-available rows
inside each strict A-grade lineage.  A seven-day chronological outer block
generates OOF hurdle scores.  Each block's isotonic EV map is fit only on
earlier *OOF-scored*, label-resolved rows from that same lineage and arm.

`1m` bars are nested in the exact 12-hour label/replay only.  This runner
never creates minute candidates and never pools the 2025 and 2026 lineages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
ID = ("candidate_id", "side_name", "__symbol__", "__ts__")
SCORE = "score_residual_expected_ev"
ALPHA = "score_base_alpha"
NET, GROSS, COST = "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"
TOP = 0.10
# Fourteen-day blocks retain the chronological OOF contract while avoiding a
# large number of nearly-identical tree refits on the same fixed cohorts.
BLOCK_DAYS, MIN_TRAIN, MIN_MAP = 14, 5_000, 2_000
MIN_STRICT_FORWARD_ROWS, MIN_STRICT_FORWARD_MONTHS = 10_000, 2
SCHEMA = "a_grade_cost_clearing_conversion_ablation_v2"
# v4 is a useful diagnostic, but predates resumable fold checkpoints.  Never
# overwrite it: v5 is the independently sealed, restart-safe rerun.
OUT = ART / "a_grade_cost_clearing_conversion_ablation_20260730_v5"

EXACT_2025 = ART / "marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet"
EXACT_2026 = ART / "mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet"
CONTEXT_2025 = ART / "canonical_execution_reliability_input_20260730_v4/panel.parquet"
REGIME_2026 = ART / "causal_execution_ev_regime_diagnostic_may_july19_20260726_v2/weekly_forward_regime_state_rows.parquet"
TRANSITION_2026 = ART / "execution_ev_transition_context_overlay_20260726_v1/strict_oof_transition_context_predictions.parquet"

REGIME_FEATURES_2025 = (
    "__regime_source_shock_impulse_score__", "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__", "__regime_source_oi_agreement_score__",
    "__regime_source_compression_score__", "__regime_source_loud_breakout_impulse_score__",
    "__regime_source_dirty_shock_avoid_score__", "__regime_source_clean_execution_context_score__",
)
TRANSITION_FEATURES_2025 = (
    "preentry_transition__range_24h_pct__delta_3h", "preentry_transition__range_24h_pct__delta_12h",
    "preentry_transition__meta_raw__volatility_zscore__delta_3h", "preentry_transition__meta_raw__volatility_zscore__delta_12h",
    "preentry_transition__trend_r2_24__delta_3h", "preentry_transition__trend_r2_24__delta_12h",
    "preentry_transition__jump_intensity__delta_3h", "preentry_transition__jump_intensity__delta_12h",
    "preentry_transition__meta_raw__chop_score__delta_3h", "preentry_transition__meta_raw__chop_score__delta_12h",
)
REGIME_FEATURES_2026 = (
    # Posterior component IDs/dimension vary across weekly frozen fits, so
    # only identity-free uncertainty/support geometry is admissible here.
    "causal_regime_entropy",
    "causal_regime_top2_margin", "causal_regime_nearest_distance2", "causal_regime_ood_z",
    "causal_regime_distance_percentile", "causal_regime_distance_exceedance",
)
TRANSITION_FEATURES_2026 = (
    "transition_probability_h1", "transition_probability_h3", "transition_probability_h6", "transition_probability_h12",
    "transition_uncertainty_h1", "transition_uncertainty_h3", "transition_uncertainty_h6", "transition_uncertainty_h12",
    "transition_probability_mean", "transition_probability_max", "transition_probability_range",
    "transition_uncertainty_mean", "transition_abstention_risk",
)
ARMS = ("baseline_residual_ev", "hurdle_alpha", "hurdle_alpha_regime", "hurdle_alpha_transition", "hurdle_alpha_regime_transition")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def stable_top(frame: pd.DataFrame, score: str) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    n = int(math.ceil(len(frame) * TOP))
    chosen = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").index[:n]
    selected.loc[chosen] = True
    return selected


def _assert_exact(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = set(ID + (SCORE, ALPHA, GROSS, COST, NET, "execution_label_end_utc"))
    missing = sorted(required.difference(frame.columns))
    if missing: raise ValueError(f"{source} missing {missing}")
    work = frame.copy(); work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["execution_label_end_utc"] = pd.to_datetime(work["execution_label_end_utc"], utc=True, errors="raise")
    if work.duplicated(list(ID)).any() or work.candidate_id.duplicated().any(): raise ValueError(f"{source} identity is not 1:1")
    if not (work.__ts__.dt.minute.eq(0) & work.__ts__.dt.second.eq(0)).all(): raise ValueError(f"{source} is not hourly")
    if not np.allclose(work[GROSS] - work[COST], work[NET], atol=1e-10, rtol=0.0): raise ValueError(f"{source} gross-cost-net mismatch")
    if (work.execution_label_end_utc <= work.__ts__).any(): raise ValueError(f"{source} outcome availability invalid")
    return work


def _join_exact_context(exact: pd.DataFrame, context: pd.DataFrame, columns: Iterable[str], *, source: str) -> pd.DataFrame:
    columns = list(columns)
    missing = sorted(set(ID + tuple(columns)).difference(context.columns))
    if missing: raise ValueError(f"{source} context missing {missing}")
    right = context.loc[:, [*ID, *columns]].copy()
    right["__ts__"] = pd.to_datetime(right["__ts__"], utc=True, errors="raise")
    if right.duplicated(list(ID)).any(): raise ValueError(f"{source} duplicate context identities")
    joined = exact.merge(right, on=list(ID), how="inner", validate="one_to_one")
    if joined.empty: raise ValueError(f"{source} empty exact/context intersection")
    if joined.loc[:, columns].isna().any().any(): raise ValueError(f"{source} context has missing values after exact join")
    return joined


def load_lineages() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    exact25 = _assert_exact(pd.read_parquet(EXACT_2025), "2025 strict A-grade")
    ctx25 = pd.read_parquet(CONTEXT_2025)
    # These context families are explicitly pre-entry.  No context target,
    # action, mapped score, or outcome-derived state is admitted.
    x25 = _join_exact_context(exact25, ctx25, [*REGIME_FEATURES_2025, *TRANSITION_FEATURES_2025], source="2025")
    x25["lineage_id"] = "canonical_marapr2025_strict_residual_oof_context_common"
    x25["evidence_grade"] = "A_STRICT_OOF_EXACT_POLICY"
    x25["regime_columns"] = "|".join(REGIME_FEATURES_2025); x25["transition_columns"] = "|".join(TRANSITION_FEATURES_2025)

    exact26 = _assert_exact(pd.read_parquet(EXACT_2026), "2026 strict A-grade")
    regime = pd.read_parquet(REGIME_2026)
    # The weekly forward state is only usable after its disclosed frozen fit
    # cutoff.  Future-change labels and their availability timestamps are not
    # in REGIME_2026 and cannot become inputs.
    regime["__ts__"] = pd.to_datetime(regime["__ts__"], utc=True, errors="raise")
    regime = regime.loc[pd.to_datetime(regime["regime_fit_cutoff_utc"], utc=True, errors="raise") <= regime["__ts__"]].copy()
    trans = pd.read_parquet(TRANSITION_2026)
    trans = trans.loc[trans["evaluation_origin"].eq("historical_outer_oof")].copy()
    x26 = _join_exact_context(exact26, regime, REGIME_FEATURES_2026, source="2026 regime")
    x26 = _join_exact_context(x26, trans, TRANSITION_FEATURES_2026, source="2026 transition")
    x26["lineage_id"] = "current_mayjul2026_strict_residual_oof_context_common"
    x26["evidence_grade"] = "A_STRICT_OOF_EXACT_POLICY"
    x26["regime_columns"] = "|".join(REGIME_FEATURES_2026); x26["transition_columns"] = "|".join(TRANSITION_FEATURES_2026)
    source = {
        x25.lineage_id.iloc[0]: {"exact": EXACT_2025, "context": CONTEXT_2025, "context_start": x25.__ts__.min(), "context_end": x25.__ts__.max()},
        x26.lineage_id.iloc[0]: {"exact": EXACT_2026, "regime": REGIME_2026, "transition": TRANSITION_2026, "context_start": x26.__ts__.min(), "context_end": x26.__ts__.max()},
    }
    return {x25.lineage_id.iloc[0]: x25, x26.lineage_id.iloc[0]: x26}, source


def features(frame: pd.DataFrame, arm: str, regime: list[str], transition: list[str]) -> np.ndarray:
    base = [SCORE, ALPHA]
    if arm in ("hurdle_alpha_regime", "hurdle_alpha_regime_transition"): base += regime
    if arm in ("hurdle_alpha_transition", "hurdle_alpha_regime_transition"): base += transition
    values = frame.loc[:, base].apply(pd.to_numeric, errors="raise").to_numpy(float)
    return np.column_stack([values, frame.side_name.eq("long").astype(float).to_numpy()])


def hurdle_score(train: pd.DataFrame, test: pd.DataFrame, arm: str, regime: list[str], transition: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    if arm == "baseline_residual_ev":
        return test[SCORE].to_numpy(float), {"model": "frozen_residual_expected_ev", "positive_rate_train": None}
    x_train, x_test = features(train, arm, regime, transition), features(test, arm, regime, transition)
    target = train[NET].gt(0).astype(int).to_numpy()
    if target.min() == target.max():
        probability = np.full(len(test), float(target.mean()))
    else:
        # Fixed, regularised logistic hurdle: deliberately small/fast so the
        # ablation tests the conversion contract rather than tree HPO.
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, max_iter=300, class_weight="balanced", random_state=20260730))
        model.fit(x_train, target); probability = model.predict_proba(x_test)[:, 1]
    # Payoffs are estimated from training labels only, separately by side.
    payoff_by_side = {}
    all_pos = train.loc[train[NET].gt(0), NET]
    all_neg = train.loc[train[NET].le(0), NET]
    for side in test.side_name.drop_duplicates():
        local = train.loc[train.side_name.eq(side), NET]
        pos, neg = local.loc[local.gt(0)], local.loc[local.le(0)]
        good = float(pos.mean()) if len(pos) else float(all_pos.mean())
        bad = float(neg.mean()) if len(neg) else float(all_neg.mean())
        payoff_by_side[str(side)] = (good, bad)
    payoff = np.asarray([payoff_by_side[str(side)] for side in test.side_name])
    score = probability * payoff[:, 0] + (1.0 - probability) * payoff[:, 1]
    return score, {"model": "blocked_standardized_logistic_cost_clear_classifier_plus_train_side_payoff", "positive_rate_train": float(target.mean())}


def causal_map(prior: pd.DataFrame, score: np.ndarray, start: pd.Timestamp) -> tuple[np.ndarray, dict[str, Any]]:
    reference = prior.loc[pd.to_datetime(prior.execution_label_end_utc, utc=True) < start].copy()
    if len(reference) < MIN_MAP or reference.raw_score.nunique() < 2:
        return np.full(len(score), np.nan), {"map_eligible": False, "map_reference_rows": int(len(reference))}
    mapper = IsotonicRegression(out_of_bounds="clip")
    mapper.fit(reference.raw_score.to_numpy(float), reference[NET].to_numpy(float))
    return mapper.predict(score), {"map_eligible": True, "map_reference_rows": int(len(reference))}


def lineage_key(frame: pd.DataFrame) -> str:
    """Filesystem-safe stable lineage key, deliberately not an opaque hash."""
    value = str(frame.lineage_id.iloc[0])
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def block_starts(frame: pd.DataFrame) -> list[pd.Timestamp]:
    origin = frame.__ts__.min().floor("D")
    blocks = origin + pd.to_timedelta(
        ((frame.__ts__.dt.normalize() - origin).dt.days // BLOCK_DAYS) * BLOCK_DAYS,
        unit="D",
    )
    return sorted(pd.DatetimeIndex(blocks.unique()).tolist())


def checkpoint_path(checkpoints: Path, frame: pd.DataFrame, block: pd.Timestamp) -> Path:
    return checkpoints / lineage_key(frame) / f"block_{block.strftime('%Y%m%dT%H%M%SZ')}.parquet"


def checkpoint_audit_path(checkpoints: Path, frame: pd.DataFrame, block: pd.Timestamp) -> Path:
    return checkpoints / lineage_key(frame) / f"block_{block.strftime('%Y%m%dT%H%M%SZ')}.audit.json"


def input_identity_sha(frame: pd.DataFrame) -> str:
    """Small but exact identity/fold guard for a checkpoint lineage.

    The full source parquet checksums remain in the final manifest.  This
    digest prevents accidentally resuming a checkpoint against a differently
    filtered exact/context intersection.
    """
    work = frame.loc[:, list(ID)].sort_values(list(ID), kind="stable").copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True).astype("int64")
    return hashlib.sha256(pd.util.hash_pandas_object(work, index=False).values.tobytes()).hexdigest()


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    table.to_parquet(partial, index=False)
    os.replace(partial, path)


def _checkpoint_audit(
    checkpoints: Path,
    frame: pd.DataFrame,
    block: pd.Timestamp,
    *,
    expected_identity_sha: str,
) -> dict[str, Any] | None:
    path = checkpoint_audit_path(checkpoints, frame, block)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("lineage_id") != frame.lineage_id.iloc[0]:
        raise ValueError(f"incompatible checkpoint audit {path}")
    if payload.get("input_identity_sha256") != expected_identity_sha:
        raise ValueError(f"checkpoint input identity changed: {path}")
    if payload.get("outer_block_start_utc") != str(block):
        raise ValueError(f"checkpoint block mismatch: {path}")
    return payload


def _load_prior_checkpoint_scores(
    checkpoints: Path,
    frame: pd.DataFrame,
    block: pd.Timestamp,
    *,
    expected_identity_sha: str,
) -> dict[str, pd.DataFrame]:
    """Load all *sealed earlier* score blocks once, then partition by arm.

    This is the lookup fix: score-lineage used to repeatedly concatenate every
    preceding arm ledger inside the inner arm loop.  Here the small set of
    earlier checkpoint files is read once per requested block and each arm
    receives a vectorised view.  An incomplete prior fold is a hard error,
    rather than an implicit changed calibration sample.
    """
    tables: list[pd.DataFrame] = []
    for earlier in block_starts(frame):
        if earlier >= block:
            break
        audit = _checkpoint_audit(checkpoints, frame, earlier, expected_identity_sha=expected_identity_sha)
        if audit is None:
            raise ValueError(
                f"cannot score {block}: required earlier checkpoint {earlier} is absent; "
                "run folds in chronological order or use --resume"
            )
        if audit["status"] == "scored":
            scored_path = checkpoint_path(checkpoints, frame, earlier)
            if not scored_path.exists():
                raise ValueError(f"checkpoint audit is scored but payload is absent: {scored_path}")
            tables.append(pd.read_parquet(scored_path))
        elif audit["status"] != "warmup_unscored":
            raise ValueError(f"unknown checkpoint status in {checkpoint_audit_path(checkpoints, frame, earlier)}")
    empty = pd.DataFrame(columns=["execution_label_end_utc", "raw_score", NET])
    if not tables:
        return {arm: empty.copy() for arm in ARMS}
    prior = pd.concat(tables, ignore_index=True)
    prior["execution_label_end_utc"] = pd.to_datetime(prior["execution_label_end_utc"], utc=True, errors="raise")
    return {
        arm: prior.loc[prior.arm.eq(arm), ["execution_label_end_utc", "raw_score", NET]].copy()
        for arm in ARMS
    }


def _write_checkpoint_audit(checkpoints: Path, frame: pd.DataFrame, block: pd.Timestamp, audit: dict[str, Any]) -> None:
    path = checkpoint_audit_path(checkpoints, frame, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != safe(audit):
            raise FileExistsError(f"checkpoint audit already exists with different contents: {path}")
        return
    write_json(path, audit)


def score_fold(
    frame: pd.DataFrame,
    block: pd.Timestamp,
    *,
    checkpoints: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one chronological 14-day block and seal it as a checkpoint.

    A fold can be launched independently only after the preceding chronological
    checkpoints are sealed.  This is intentional: the map for a later fold is
    defined by earlier blocked-OOS predictions, not by a reconstructed or
    in-sample surrogate.  A completed checkpoint is immutable and is reused
    when ``resume`` is true.
    """
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    if block not in block_starts(frame):
        raise ValueError(f"{block} is not a valid outer block for {frame.lineage_id.iloc[0]}")
    identity_sha = input_identity_sha(frame)
    existing = _checkpoint_audit(checkpoints, frame, block, expected_identity_sha=identity_sha)
    payload_path = checkpoint_path(checkpoints, frame, block)
    if existing is not None:
        if not resume:
            raise FileExistsError(f"checkpoint exists; pass --resume: {payload_path}")
        if existing["status"] == "warmup_unscored":
            return pd.DataFrame(), pd.DataFrame([existing])
        if not payload_path.exists():
            raise ValueError(f"sealed scored checkpoint has no payload: {payload_path}")
        return pd.read_parquet(payload_path), pd.DataFrame([existing])

    test = frame.loc[frame.__ts__.ge(block) & frame.__ts__.lt(block + pd.Timedelta(days=BLOCK_DAYS))].copy()
    train = frame.loc[(frame.__ts__ < block) & (frame.execution_label_end_utc < block)].copy()
    started = time.perf_counter()
    common = {
        "schema": SCHEMA,
        "lineage_id": frame.lineage_id.iloc[0],
        "outer_block_start_utc": str(block),
        "input_identity_sha256": identity_sha,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "candidate_cadence": "1h",
        "label_resolution_gate": "execution_label_end_utc < outer_block_start_utc",
    }
    if len(train) < MIN_TRAIN:
        audit = {**common, "status": "warmup_unscored", "wall_seconds": time.perf_counter() - started}
        _write_checkpoint_audit(checkpoints, frame, block, audit)
        return pd.DataFrame(), pd.DataFrame([audit])

    # Read all earlier blocks once.  The lookup is then one arm-filter per
    # fitted arm, not a growing concat in the arm loop.
    prior_by_arm = _load_prior_checkpoint_scores(
        checkpoints, frame, block, expected_identity_sha=identity_sha,
    )
    regime, transition = frame.regime_columns.iloc[0].split("|"), frame.transition_columns.iloc[0].split("|")
    ledgers: list[pd.DataFrame] = []
    arm_audit: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_started = time.perf_counter()
        raw, model = hurdle_score(train, test, arm, regime, transition)
        mapped, mapping = causal_map(prior_by_arm[arm], raw, block)
        output = test.loc[:, [*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET]].copy()
        output["arm"] = arm
        output["outer_block_start_utc"] = block
        output["raw_score"] = raw
        output["mapped_ev"] = mapped
        output["map_eligible"] = mapping["map_eligible"]
        output["map_reference_rows"] = mapping["map_reference_rows"]
        ledgers.append(output)
        arm_audit.append({
            **common, "status": "scored", "arm": arm, **model, **mapping,
            "arm_wall_seconds": time.perf_counter() - arm_started,
        })
    ledger = pd.concat(ledgers, ignore_index=True)
    _atomic_parquet(ledger, payload_path)
    audit = {
        **common,
        "status": "scored",
        "arm_audit": arm_audit,
        "payload_sha256": sha(payload_path),
        "payload_rows": int(len(ledger)),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_checkpoint_audit(checkpoints, frame, block, audit)
    return ledger, pd.DataFrame(arm_audit)


def score_lineage(
    frame: pd.DataFrame,
    *,
    checkpoints: Path | None = None,
    resume: bool = True,
    only_block: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score/materialise a lineage in deterministic chronological batches.

    ``only_block`` makes a single 14-day fold invocation practical.  The
    default runs missing folds in order and resumes sealed checkpoints.  This
    function deliberately returns only scored (not warmup) rows.
    """
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    checkpoints = checkpoints or (OUT / "fold_checkpoints")
    starts = block_starts(frame)
    if only_block is not None:
        block = pd.Timestamp(only_block)
        if block.tzinfo is None: block = block.tz_localize("UTC")
        else: block = block.tz_convert("UTC")
        ledger, audit = score_fold(frame, block, checkpoints=checkpoints, resume=resume)
        return ledger, audit
    ledgers: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for block in starts:
        ledger, audit = score_fold(frame, block, checkpoints=checkpoints, resume=resume)
        if not ledger.empty: ledgers.append(ledger)
        audits.append(audit)
    empty = pd.DataFrame(columns=[*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET, "arm", "outer_block_start_utc", "raw_score", "mapped_ev", "map_eligible", "map_reference_rows"])
    return (pd.concat(ledgers, ignore_index=True) if ledgers else empty,
            pd.concat(audits, ignore_index=True) if audits else pd.DataFrame())


def rank_ic(frame: pd.DataFrame, score: str, target: str) -> float:
    if len(frame) < 2 or frame[score].nunique() < 2 or frame[target].nunique() < 2: return float("nan")
    return float(frame[score].corr(frame[target], method="spearman"))


def metrics(frame: pd.DataFrame, *, period_type: str, period: str, selected: bool, candidate_rows: int) -> dict[str, Any]:
    values = frame[NET]
    daily = frame.groupby(frame.__ts__.dt.floor("D"), observed=True)[NET].mean()
    assets = frame.__symbol__.value_counts(normalize=True)
    return {"period_type": period_type, "period": period, "candidate_rows": candidate_rows, "selected_rows": len(frame) if selected else None,
            "raw_score_net_rank_ic": rank_ic(frame, "raw_score", NET), "mapped_ev_net_rank_ic": rank_ic(frame, "mapped_ev", NET),
            "mean_gross_bps": float(frame[GROSS].mean()*1e4), "mean_cost_bps": float(frame[COST].mean()*1e4), "mean_net_bps": float(values.mean()*1e4),
            "net_trade_q10_bps": float(values.quantile(.10)*1e4), "net_trade_q50_bps": float(values.quantile(.50)*1e4),
            "net_day_q10_bps": float(daily.quantile(.10)*1e4), "net_day_q50_bps": float(daily.quantile(.50)*1e4),
            "positive_net_rate": float(values.gt(0).mean()), "long_share": float(frame.side_name.eq("long").mean()),
            "asset_count": int(frame.__symbol__.nunique()), "largest_asset_share": float(assets.iloc[0]) if len(assets) else float("nan"),
            "asset_hhi": float((assets**2).sum()) if len(assets) else float("nan")}


def evaluate(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = scored.loc[scored.map_eligible].copy()
    eligible["month"] = eligible.__ts__.dt.strftime("%Y-%m"); eligible["week"] = eligible.__ts__.dt.strftime("%G-W%V")
    period_rows=[]; selected_rows=[]
    for (lineage, arm, month), group in eligible.groupby(["lineage_id", "arm", "month"], sort=True, observed=True):
        mask = stable_top(group, "mapped_ev")
        selected = group.loc[mask].copy(); selected["selection_month"] = month; selected_rows.append(selected)
        common={"lineage_id":lineage,"evidence_grade":group.evidence_grade.iloc[0],"arm":arm,"selection_scope":"one_pooled_global_top10_after_arm_local_causal_mapping"}
        period_rows.append({**common, **metrics(group, period_type="month", period=month, selected=False, candidate_rows=len(group))})
        period_rows.append({**common, **metrics(selected, period_type="month", period=month, selected=True, candidate_rows=len(group))})
        for side, local in selected.groupby("side_name", sort=True, observed=True): period_rows.append({**common,"side_name":side,**metrics(local, period_type="month", period=month, selected=True,candidate_rows=len(group))})
    selected_all = pd.concat(selected_rows, ignore_index=True)
    # Weeks retain the monthly pooled-book membership; they never re-rank.
    for (lineage, arm, week), group in selected_all.groupby(["lineage_id", "arm", selected_all.__ts__.dt.strftime("%G-W%V")], sort=True, observed=True):
        common={"lineage_id":lineage,"evidence_grade":group.evidence_grade.iloc[0],"arm":arm,"selection_scope":"monthly_pooled_global_top10_membership_viewed_by_week"}
        population = int(eligible.loc[(eligible.lineage_id.eq(lineage))&(eligible.arm.eq(arm))&(eligible.week.eq(week))].shape[0])
        period_rows.append({**common,**metrics(group,period_type="week",period=week,selected=True,candidate_rows=population)})
        for side, local in group.groupby("side_name",sort=True,observed=True):period_rows.append({**common,"side_name":side,**metrics(local,period_type="week",period=week,selected=True,candidate_rows=population)})
    periods=pd.DataFrame(period_rows)
    recall=[]
    for (lineage,arm,month), group in eligible.groupby(["lineage_id","arm","month"],sort=True,observed=True):
        selected=selected_all.loc[(selected_all.lineage_id.eq(lineage))&(selected_all.arm.eq(arm))&(selected_all.selection_month.eq(month))]
        for label,mask in (("positive_net",group[NET].gt(0)),("gross_exceeds_cost",group[GROSS].gt(group[COST]))):
            denom=int(mask.sum())
            # Identity membership avoids relying on index values after concat.
            ids=set(selected.candidate_id); hit=int(group.loc[mask,"candidate_id"].isin(ids).sum())
            recall.append({"lineage_id":lineage,"arm":arm,"month":month,"event":label,"population_event_rows":denom,"selected_event_rows":hit,"recall":hit/denom if denom else float("nan")})
    summaries=[]
    for (lineage,arm,ptype), group in periods.loc[periods.selected_rows.notna() & periods.side_name.isna() if "side_name" in periods else periods.selected_rows.notna()].groupby(["lineage_id","arm","period_type"],sort=True,observed=True):
        latest=group.loc[group.period.eq(group.period.max())].iloc[0]; worst=group.loc[group.mean_net_bps.idxmin()]
        summaries.append({"lineage_id":lineage,"arm":arm,"period_type":ptype,"periods":int(group.period.nunique()),"aggregate_mean_net_bps":float(group.mean_net_bps.mean()),"latest_period":latest.period,"latest_mean_net_bps":float(latest.mean_net_bps),"worst_period":worst.period,"worst_mean_net_bps":float(worst.mean_net_bps),"aggregate_q10_bps":float(group.net_trade_q10_bps.quantile(.10)),"aggregate_q50_bps":float(group.net_trade_q50_bps.quantile(.50))})
    return periods, pd.DataFrame(recall), pd.DataFrame(summaries)


def frozen_2025_to_2026(
    historical: pd.DataFrame, historical_oof: pd.DataFrame, current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen common-feature forward comparison; never touches a 2026 label to fit.

    Regime/transition arms intentionally do not appear here.  The supplied
    2025 and 2026 context sidecars use different feature contracts, so carrying
    a 2025 context model into 2026 would be undefined rather than a strict
    forward test.  They remain separately labelled within-lineage diagnostics.
    """
    months = current.__ts__.dt.strftime("%Y-%m").nunique()
    if len(current) < MIN_STRICT_FORWARD_ROWS or months < MIN_STRICT_FORWARD_MONTHS:
        raise ValueError("2026 causal-context intersection is too late/small for an honest strict forward comparison")
    regime25, transition25 = historical.regime_columns.iloc[0].split("|"), historical.transition_columns.iloc[0].split("|")
    rows=[]; availability=[]
    for arm in ("baseline_residual_ev", "hurdle_alpha"):
        raw25, model = hurdle_score(historical, current, arm, regime25, transition25)
        # Calibration sees only blocked OOF predictions from 2025, never
        # in-sample full-fit scores and never any 2026 realised outcome.
        calibration = historical_oof.loc[historical_oof.arm.eq(arm), ["raw_score", NET]].dropna()
        if len(calibration) < MIN_MAP or calibration.raw_score.nunique() < 2:
            raise ValueError(f"insufficient 2025 blocked-OOF calibration support for {arm}")
        mapper=IsotonicRegression(out_of_bounds="clip").fit(calibration.raw_score,calibration[NET])
        out=current.loc[:, [*ID,"lineage_id","evidence_grade","execution_label_end_utc",GROSS,COST,NET]].copy()
        out["arm"]=arm; out["raw_score"]=raw25; out["mapped_ev"]=mapper.predict(raw25); out["map_eligible"]=True
        out["map_reference_rows"]=len(calibration); out["outer_block_start_utc"]=pd.Timestamp("2026-01-01T00:00:00Z")
        out["evaluation_role"]="FROZEN_2025_SELECTION_CALIBRATION_TO_2026"
        rows.append(out); availability.append({"arm":arm,"status":"strict_forward_available","model":model["model"],"fit_labels":"2025 only","map_labels":"2025 blocked OOF only","2026_rows":len(out),"2025_oof_map_rows":len(calibration)})
    for arm in ("hurdle_alpha_regime","hurdle_alpha_transition","hurdle_alpha_regime_transition"):
        availability.append({"arm":arm,"status":"fail_closed_noncomparable_2025_2026_context_feature_contract","model":None,"fit_labels":"not fit","map_labels":"not fit","2026_rows":0,"2025_oof_map_rows":0})
    return pd.concat(rows,ignore_index=True), pd.DataFrame(availability)


def _select_lineages(lineages: dict[str, pd.DataFrame], which: str) -> dict[str, pd.DataFrame]:
    if which == "all":
        return lineages
    prefix = "canonical_marapr" if which == "2025" else "current_mayjul"
    selected = {name: frame for name, frame in lineages.items() if name.startswith(prefix)}
    if len(selected) != 1:
        raise ValueError(f"could not select exactly one {which} lineage")
    return selected


def _flatten_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Keep one profiler/audit row per arm and one row for warmup blocks."""
    if audit.empty:
        return audit
    # score_fold already returns one row per arm.  This helper keeps an
    # explicit guard for callers that obtain a checkpoint-level audit later.
    return audit.drop(columns=["arm_audit"], errors="ignore")


def run(
    output: Path = OUT,
    *,
    checkpoints: Path | None = None,
    lineage: str = "all",
    only_block: pd.Timestamp | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Materialise the complete sealed diagnostic or one resumable fold.

    A one-fold invocation intentionally stops after its checkpoint.  It does
    not fabricate a partial final verdict.  The full invocation verifies every
    expected fold before it merges candidate ledgers and evaluates the frozen
    2025→2026 comparison.
    """
    checkpoint_root = checkpoints or (output.parent / f"{output.name}_fold_checkpoints")
    if only_block is not None and lineage == "all":
        raise ValueError("--outer-block-start requires --lineage 2025 or 2026")
    if only_block is not None:
        lineages, _ = load_lineages()
        selected = _select_lineages(lineages, lineage)
        frame = next(iter(selected.values()))
        ledger, audit = score_lineage(
            frame, checkpoints=checkpoint_root, resume=resume, only_block=only_block,
        )
        return {
            "schema": SCHEMA,
            "status": "SEALED_FOLD_CHECKPOINT_ONLY",
            "promotion_eligible": False,
            "lineage_id": frame.lineage_id.iloc[0],
            "outer_block_start_utc": str(pd.Timestamp(only_block)),
            "checkpoint_root": str(checkpoint_root),
            "candidate_rows": int(len(ledger)),
            "audit_rows": int(len(audit)),
        }
    if output.exists(): raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        all_lineages, source = load_lineages()
        # A final verdict necessarily needs both lineages.  Retain a separate
        # one-lineage checkpoint command above for slow/restart-prone hosts.
        if lineage != "all":
            raise ValueError("a final strict-forward verdict requires --lineage all")
        lineages = all_lineages
        scores=[]; audits=[]; coverage=[]
        for lineage_id, frame in lineages.items():
            coverage.append({"lineage_id":lineage_id,"evidence_grade":frame.evidence_grade.iloc[0],"strict_a_grade_rows":int(len(pd.read_parquet(source[lineage_id]["exact"]))),"context_common_rows":len(frame),"context_start_utc":frame.__ts__.min(),"context_end_utc":frame.__ts__.max(),"candidate_cadence":"1h","checkpoint_root":str(checkpoint_root),"folds_expected":len(block_starts(frame))})
            ledger, audit=score_lineage(frame, checkpoints=checkpoint_root, resume=resume); scores.append(ledger); audits.append(_flatten_audit(audit))
        scored=pd.concat(scores,ignore_index=True); audit=pd.concat(audits,ignore_index=True)
        periods, recall, summary=evaluate(scored)
        historical_name=[name for name in lineages if name.startswith("canonical_marapr")][0]
        current_name=[name for name in lineages if name.startswith("current_mayjul")][0]
        strict_scores, strict_availability=frozen_2025_to_2026(lineages[historical_name], scored.loc[scored.lineage_id.eq(historical_name)], lineages[current_name])
        strict_periods, strict_recall, strict_summary=evaluate(strict_scores)
        for table in (scored, strict_scores):
            table["evaluation_role"]=table.get("evaluation_role", "WITHIN_LINEAGE_CHRONOLOGICAL_OOF_DIAGNOSTIC")
        for name,table in (("context_coverage.csv",pd.DataFrame(coverage)),("blocked_oof_fit_mapping_audit.csv",audit),("within_lineage_candidate_scores.parquet",scored),("within_lineage_period_metrics.csv",periods),("within_lineage_recall.csv",recall),("within_lineage_summary.csv",summary),("strict_forward_2026_candidate_scores.parquet",strict_scores),("strict_forward_2026_period_metrics.csv",strict_periods),("strict_forward_2026_recall.csv",strict_recall),("strict_forward_2026_summary.csv",strict_summary),("strict_forward_arm_availability.csv",strict_availability)):
            (table.to_parquet(staging/name,index=False) if name.endswith(".parquet") else table.to_csv(staging/name,index=False))
        inputs={str(p):sha(p) for p in (EXACT_2025,EXACT_2026,CONTEXT_2025,REGIME_2026,TRANSITION_2026)}
        checkpoint_audits = []
        for frame in lineages.values():
            identity_sha = input_identity_sha(frame)
            for block in block_starts(frame):
                item = _checkpoint_audit(checkpoint_root, frame, block, expected_identity_sha=identity_sha)
                if item is None:
                    raise ValueError(f"missing checkpoint after full lineage run: {frame.lineage_id.iloc[0]} {block}")
                checkpoint_audits.append(item)
        profile = {
            "fold_checkpoint_count": len(checkpoint_audits),
            "scored_checkpoint_count": sum(item["status"] == "scored" for item in checkpoint_audits),
            "warmup_checkpoint_count": sum(item["status"] == "warmup_unscored" for item in checkpoint_audits),
            "total_checkpoint_wall_seconds": float(sum(float(item.get("wall_seconds", 0.0)) for item in checkpoint_audits)),
            "maximum_checkpoint_wall_seconds": float(max(float(item.get("wall_seconds", 0.0)) for item in checkpoint_audits)),
            "lookup_contract": "earlier checkpoint payloads loaded once per block; vectorised partition by arm; no growing in-loop ledger concat",
        }
        write_json(staging/"checkpoint_profile.json", profile)
        report={"schema":SCHEMA,"status":"SEALED_DIAGNOSTIC_NON_PROMOTION","promotion_eligible":False,"input_sha256":inputs,
                "contract":{"candidate_cadence":"1h; 1m only nested in exact 12h label paths","lineages":"A-grade 2025/2026 separately; no cross-lineage metric or PnL pooling","context":"baseline and all arms are restricted to the same exact identity intersection per lineage","within_lineage_oof":"fourteen-day chronological blocks, resumable as immutable per-fold checkpoints; each train label resolves strictly before test block start; diagnostic only","strict_2026":"fit/select/calibrate on 2025 only; apply frozen to 2026 with no 2026 labels; only common baseline/alpha feature contract is eligible","mapping":"per-arm isotonic EV maps use only earlier blocked-OOF score/outcome rows resolved strictly before each block","selection":"one pooled global top10 per lineage/month after arm-local mapping; weekly tables retain monthly membership and never re-rank","arms":list(ARMS),"hurdle":"positive exact net (gross > explicit cost) classifier plus train-only side-conditional payoff","context_arm_forward":"regime/transition arms are explicit fail-closed because 2025/2026 sidecars have incompatible feature semantics"},
                "checkpoint_profile":profile,
                "outputs_sha256":{p.name:sha(p) for p in staging.iterdir() if p.is_file()}}
        write_json(staging/"report.json",report)
        manifest={"schema":SCHEMA+"_manifest","status":"SEALED_DIAGNOSTIC_NON_PROMOTION","runner_sha256":sha(Path(__file__)),"input_sha256":inputs,"outputs_sha256":{p.name:sha(p) for p in staging.iterdir() if p.is_file()}}
        write_json(staging/"manifest.json",manifest); (staging/"manifest.sha256").write_text(f"{sha(staging/'manifest.json')}  manifest.json\n")
        staging.replace(output); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--checkpoints",type=Path,default=None,help="persistent immutable 14-day OOF fold checkpoint root")
    parser.add_argument("--lineage",choices=("all","2025","2026"),default="all")
    parser.add_argument("--outer-block-start",type=pd.Timestamp,default=None,help="run/resume one UTC 14-day fold only; requires --lineage")
    parser.add_argument("--no-resume",action="store_true",help="fail if any requested fold checkpoint already exists")
    args=parser.parse_args()
    print(json.dumps(safe(run(args.output, checkpoints=args.checkpoints, lineage=args.lineage, only_block=args.outer_block_start, resume=not args.no_resume)),indent=2))
if __name__ == "__main__": main()
