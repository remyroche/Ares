"""Live threshold-basis policy transforms for portfolio inference.

The promoted policy is intentionally separated from the historical hit-rate
surprise layer. It turns a batch of live decisions into the same top-band rank
space used by the offline threshold-basis replay: historical prior rows define
reachable EV thresholds, while current rows are selected per timestamp batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


_REFERENCE_CACHE: dict[str, pd.DataFrame] = {}


def load_threshold_basis_policy(path: str | Path | None) -> dict[str, Any]:
    """Load a threshold-basis policy artifact."""
    if path is None:
        return {}
    policy_path = Path(path)
    if not policy_path.exists():
        return {}
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    payload["_artifact_path"] = str(policy_path)
    return payload


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _first_non_empty(mapping: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "<na>"}:
            return text
    return ""


def _policy_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = str(payload.get(key) or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    artifact_path = str(payload.get("_artifact_path") or "").strip()
    if artifact_path:
        candidate = Path(artifact_path).parent / path
        if candidate.exists():
            return candidate
    return path


def _load_reference(payload: Mapping[str, Any]) -> pd.DataFrame:
    path = _policy_path(payload, "reference_candidates_path")
    if path is None or not path.exists():
        return pd.DataFrame()
    key = str(path.resolve())
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached
    columns = payload.get("reference_columns")
    kwargs: dict[str, Any] = {}
    if isinstance(columns, Iterable) and not isinstance(columns, (str, bytes)):
        kwargs["columns"] = list(columns)
    ref = pd.read_parquet(path, **kwargs)
    ref = _normalise_rows(ref)
    _REFERENCE_CACHE[key] = ref
    return ref


def _normalise_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "policy_archetype" not in out.columns:
        for col in ("local_side_archetype", "archetype_label_family", "source_archetype"):
            if col in out.columns:
                out["policy_archetype"] = out[col].astype(str)
                break
    if "policy_archetype" not in out.columns:
        out["policy_archetype"] = "missing"
    return out


def _rank_against_ref(score: pd.Series, ref_score: pd.Series) -> pd.Series:
    ref = (
        pd.to_numeric(ref_score, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ref_arr = np.sort(ref.to_numpy(dtype=np.float64, copy=False))
    out = np.full(len(score), np.nan, dtype=np.float64)
    if ref_arr.size == 0:
        return pd.Series(out, index=score.index)
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    out[finite] = (
        np.searchsorted(ref_arr, values[finite], side="right") / float(ref_arr.size)
    )
    return pd.Series(out, index=score.index).clip(0.0, 1.0)


def _threshold_for_target_ev(
    ref: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    target_ev: float,
    min_rows: int,
) -> float:
    if ref.empty or score_col not in ref.columns or return_col not in ref.columns:
        return float("nan")
    score = pd.to_numeric(ref[score_col], errors="coerce")
    ev = pd.to_numeric(ref[return_col], errors="coerce")
    valid = ref.loc[score.notna() & ev.notna(), [score_col, return_col]].copy()
    if len(valid) < int(min_rows) or not np.isfinite(float(target_ev)):
        return float("nan")
    values = pd.to_numeric(valid[score_col], errors="coerce")
    grid = np.unique(
        np.nanquantile(values.to_numpy(dtype=float), np.linspace(0.70, 0.99, 60))
    )
    best_threshold = float("nan")
    best_gap = float("inf")
    for threshold in grid:
        chosen = valid.loc[pd.to_numeric(valid[score_col], errors="coerce").ge(float(threshold))]
        if len(chosen) < int(min_rows):
            continue
        mean_ev = float(pd.to_numeric(chosen[return_col], errors="coerce").mean())
        gap = abs(mean_ev - float(target_ev))
        if mean_ev >= float(target_ev) and gap < best_gap:
            best_gap = gap
            best_threshold = float(threshold)
    if not np.isfinite(best_threshold):
        best_threshold = float(np.nanquantile(values.to_numpy(dtype=float), 0.99))
    return best_threshold


def _historical_top10_ev(
    ref: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    min_rows: int,
) -> float:
    if ref.empty or score_col not in ref.columns or return_col not in ref.columns:
        return float("nan")
    score = pd.to_numeric(ref[score_col], errors="coerce")
    ev = pd.to_numeric(ref[return_col], errors="coerce")
    valid = ref.loc[score.notna() & ev.notna(), [score_col, return_col]]
    if len(valid) < int(min_rows):
        return float("nan")
    threshold = float(pd.to_numeric(valid[score_col], errors="coerce").quantile(0.90))
    chosen = valid.loc[pd.to_numeric(valid[score_col], errors="coerce").ge(threshold)]
    if len(chosen) < int(min_rows):
        return float("nan")
    return float(pd.to_numeric(chosen[return_col], errors="coerce").mean())


def _historical_current_top10_ev(
    ref: pd.DataFrame,
    *,
    return_col: str,
    rank_col: str,
    min_rows: int,
) -> float:
    if ref.empty or return_col not in ref.columns or rank_col not in ref.columns:
        return float("nan")
    rank = pd.to_numeric(ref[rank_col], errors="coerce")
    ev = pd.to_numeric(ref[return_col], errors="coerce")
    valid = ref.loc[rank.notna() & ev.notna()].copy()
    if len(valid) < int(min_rows):
        return float("nan")
    chosen = valid.loc[pd.to_numeric(valid[rank_col], errors="coerce").ge(0.90)]
    if len(chosen) < max(1, int(min_rows) // 2):
        return float("nan")
    return float(pd.to_numeric(chosen[return_col], errors="coerce").mean())


def _top_n_by_score(rows: pd.DataFrame, score: pd.Series, n: int) -> pd.DataFrame:
    if rows.empty or int(n) <= 0:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    work = rows.copy()
    work["_activity_score"] = pd.to_numeric(score.reindex(work.index), errors="coerce")
    work = work.loc[work["_activity_score"].notna()].copy()
    if work.empty:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    sort_cols = [col for col in ("_activity_score", "timestamp", "symbol") if col in work.columns]
    asc = [False] + [True] * (len(sort_cols) - 1)
    work = work.sort_values(sort_cols, ascending=asc).head(int(n))
    work["selection_score"] = pd.to_numeric(work["_activity_score"], errors="coerce")
    return work.drop(columns=["_activity_score"])


def _rescale_selection_score_to_top_band(
    rows: pd.DataFrame,
    *,
    floor: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    score = pd.to_numeric(out.get("selection_score"), errors="coerce")
    if score.notna().sum() == 0:
        out["selection_score"] = float(floor)
        return out
    rank = score.rank(method="first", pct=True)
    out["selection_score"] = float(floor) + (1.0 - float(floor)) * rank
    return out


def _match_baseline_activity(
    batch_rows: pd.DataFrame,
    chosen: pd.DataFrame,
    *,
    score: pd.Series,
    target_count: int,
    top_band_floor: float,
) -> pd.DataFrame:
    target = int(target_count)
    if target <= 0:
        out = batch_rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    if len(chosen) >= target:
        return _rescale_selection_score_to_top_band(
            _top_n_by_score(chosen, score, target),
            floor=top_band_floor,
        )
    return _rescale_selection_score_to_top_band(
        _top_n_by_score(batch_rows, score, target),
        floor=top_band_floor,
    )


def _score_value(decision: Mapping[str, Any], score_col: str) -> float:
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    aliases = [score_col]
    aliases.extend(
        [
            "calibrated_score_regime_ev",
            "score_regime_calibrated",
            "calibrated_score",
            "meta_score_oof",
            "meta_pred",
        ]
    )
    for name in aliases:
        if not name:
            continue
        value = decision.get(name, chain.get(name))
        out = _safe_float(value)
        if np.isfinite(out):
            return out
    return float("nan")


def _decision_timestamp(decision: Mapping[str, Any]) -> pd.Timestamp:
    for name in ("signal_bar_ts", "timestamp", "decision_ts"):
        value = decision.get(name)
        if value is None:
            continue
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return pd.Timestamp.now(tz="UTC")


def _baseline_rank(
    decision: Mapping[str, Any],
    *,
    store: Any,
) -> float:
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    for name in ("policy_rank_pct", "rank_pct"):
        value = _safe_float(decision.get(name, chain.get(name)))
        if np.isfinite(value):
            return float(np.clip(value, 0.0, 1.0))
    if store is None:
        return float("nan")
    score = _safe_float(decision.get("calibrated_score", chain.get("calibrated_score")))
    if not np.isfinite(score):
        return float("nan")
    try:
        result = store.lookup(
            strategy_id=str(decision.get("strategy_id") or ""),
            side=str(decision.get("side") or decision.get("side_name") or ""),
            calibrated_score=float(score),
        )
        value = _safe_float(getattr(result, "policy_rank_pct", np.nan))
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _decision_rows(
    decisions: Sequence[MutableMapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    store: Any,
) -> pd.DataFrame:
    live_score_col = str(
        policy.get("live_score_col")
        or policy.get("score_col")
        or "calibrated_score"
    )
    rows: list[dict[str, Any]] = []
    for idx, decision in enumerate(decisions):
        chain = decision.get("chain_results")
        if not isinstance(chain, Mapping):
            chain = {}
        side = _first_non_empty(decision, ("side_name", "side")) or _first_non_empty(chain, ("side_name", "side"))
        policy_archetype = (
            _first_non_empty(decision, ("policy_archetype", "local_side_archetype", "archetype_label_family"))
            or _first_non_empty(chain, ("policy_archetype", "local_side_archetype", "archetype_label_family"))
            or "missing"
        )
        rows.append(
            {
                "_decision_idx": idx,
                "timestamp": _decision_timestamp(decision),
                "symbol": str(decision.get("symbol") or chain.get("symbol") or ""),
                "strategy_id": str(decision.get("strategy_id") or chain.get("strategy_id") or ""),
                "side_name": str(side or ""),
                "policy_archetype": str(policy_archetype or "missing"),
                "score": _score_value(decision, live_score_col),
                "rank_pct": _baseline_rank(decision, store=store),
            }
        )
    return _normalise_rows(pd.DataFrame(rows))


def _select_batch(
    batch_rows: pd.DataFrame,
    *,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_score_col = str(
        policy.get("reference_score_col")
        or policy.get("score_col")
        or "calibrated_score_regime_ev"
    )
    return_col = str(policy.get("return_col") or "ret_net_notional")
    rank_col = str(policy.get("rank_col") or "rank_pct")
    min_rows = int(policy.get("min_reference_rows") or 40)
    arch_min_rows = int(policy.get("arch_min_reference_rows") or max(8, min_rows // 4))
    base_rank_threshold = float(policy.get("base_rank_threshold") or 0.90)
    top_band_floor = float(policy.get("top_band_floor") or base_rank_threshold)

    baseline_count = int(
        pd.to_numeric(batch_rows.get("rank_pct"), errors="coerce")
        .ge(base_rank_threshold)
        .sum()
    )
    global_target = _historical_current_top10_ev(
        all_prior,
        return_col=return_col,
        rank_col=rank_col,
        min_rows=min_rows,
    )
    if not np.isfinite(global_target):
        global_target = _historical_top10_ev(
            all_prior,
            score_col=reference_score_col,
            return_col=return_col,
            min_rows=min_rows,
        )
    global_threshold = _threshold_for_target_ev(
        recent_ref,
        score_col=reference_score_col,
        return_col=return_col,
        target_ev=global_target,
        min_rows=min_rows,
    )
    chosen_parts: list[pd.DataFrame] = []
    score_rank = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    dynamic_targets = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    dynamic_thresholds = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    for archetype, sub in batch_rows.groupby("policy_archetype", dropna=False):
        archetype_key = str(archetype)
        ref_arch = recent_ref.loc[
            recent_ref["policy_archetype"].astype(str).eq(archetype_key)
        ]
        prior_arch = all_prior.loc[
            all_prior["policy_archetype"].astype(str).eq(archetype_key)
        ]
        target = _historical_current_top10_ev(
            prior_arch,
            return_col=return_col,
            rank_col=rank_col,
            min_rows=arch_min_rows,
        )
        threshold = _threshold_for_target_ev(
            ref_arch,
            score_col=reference_score_col,
            return_col=return_col,
            target_ev=target,
            min_rows=arch_min_rows,
        )
        if not np.isfinite(threshold):
            threshold = global_threshold
            target = global_target
        dynamic_targets.loc[sub.index] = (
            float(target) if np.isfinite(float(target)) else np.nan
        )
        dynamic_thresholds.loc[sub.index] = (
            float(threshold) if np.isfinite(float(threshold)) else np.nan
        )
        ref_score = (
            ref_arch[reference_score_col]
            if len(ref_arch) >= arch_min_rows and reference_score_col in ref_arch.columns
            else recent_ref.get(reference_score_col, pd.Series(dtype="float64"))
        )
        local_rank = _rank_against_ref(sub["score"], ref_score)
        score_rank.loc[sub.index] = local_rank
        part = sub.loc[pd.to_numeric(sub["score"], errors="coerce").ge(threshold)].copy()
        part["selection_score"] = local_rank.reindex(part.index)
        chosen_parts.append(part)
    chosen = (
        pd.concat(chosen_parts, ignore_index=False)
        if chosen_parts
        else batch_rows.iloc[0:0].copy()
    )
    global_rank = _rank_against_ref(
        batch_rows["score"],
        recent_ref.get(reference_score_col, pd.Series(dtype="float64")),
    )
    selected = _match_baseline_activity(
        batch_rows,
        chosen,
        score=score_rank.fillna(global_rank),
        target_count=baseline_count,
        top_band_floor=top_band_floor,
    )
    if not selected.empty:
        selected["dynamic_ev_target"] = dynamic_targets.reindex(selected.index)
        selected["dynamic_score_threshold"] = dynamic_thresholds.reindex(selected.index)
    meta = {
        "baseline_activity_count": int(baseline_count),
        "recent_reference_rows": int(len(recent_ref)),
        "reference_rows": int(len(all_prior)),
        "global_dynamic_ev_target": float(global_target)
        if np.isfinite(global_target)
        else np.nan,
        "global_dynamic_score_threshold": float(global_threshold)
        if np.isfinite(global_threshold)
        else np.nan,
    }
    return selected, meta


def apply_threshold_basis_policy_to_decisions(
    decisions: Sequence[MutableMapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
    store: Any = None,
) -> Sequence[MutableMapping[str, Any]]:
    """Attach threshold-basis rank scores to live decisions.

    Non-selected rows receive a score of 0.0 so the regular rank threshold gate
    rejects them. Selected rows receive scores rescaled into the configured
    top band, matching the replay candidate table semantics.
    """
    if not decisions or not isinstance(policy, Mapping):
        return decisions
    enabled = bool(policy.get("enabled", True))
    if not enabled:
        return decisions
    reference = _load_reference(policy)
    if reference.empty:
        for decision in decisions:
            decision["threshold_basis_reason"] = "missing_reference_candidates"
        return decisions
    min_rows = int(policy.get("min_reference_rows") or 40)
    window_days = int(policy.get("window_days") or 8)
    family = str(policy.get("family") or "ev_target_archetype_reachable_matched_activity")
    policy_id = str(policy.get("policy_id") or policy.get("name") or "threshold_basis_policy")
    rows = _decision_rows(decisions, policy=policy, store=store)
    if rows.empty:
        return decisions
    for period, batch_rows in rows.groupby("timestamp", sort=True):
        period_ts = pd.Timestamp(period)
        ref_start = period_ts - pd.Timedelta(days=window_days)
        recent_ref = reference.loc[
            reference["timestamp"].ge(ref_start) & reference["timestamp"].lt(period_ts)
        ].copy()
        all_prior = reference.loc[reference["timestamp"].lt(period_ts)].copy()
        if len(recent_ref) < min_rows:
            for idx in batch_rows["_decision_idx"].astype(int).tolist():
                decision = decisions[idx]
                decision["threshold_basis_policy_id"] = policy_id
                decision["threshold_basis_family"] = family
                decision["threshold_basis_rank_score"] = 0.0
                decision["threshold_basis_selected"] = False
                decision["threshold_basis_reason"] = "insufficient_recent_reference"
                decision["threshold_basis_recent_reference_rows"] = int(len(recent_ref))
                decision["threshold_basis_reference_rows"] = int(len(all_prior))
            continue
        selected, meta = _select_batch(
            batch_rows,
            recent_ref=recent_ref,
            all_prior=all_prior,
            policy=policy,
        )
        selected_scores = {
            int(row["_decision_idx"]): _safe_float(row.get("selection_score"), 0.0)
            for _, row in selected.iterrows()
        }
        selected_thresholds = {
            int(row["_decision_idx"]): _safe_float(row.get("dynamic_score_threshold"))
            for _, row in selected.iterrows()
        }
        selected_targets = {
            int(row["_decision_idx"]): _safe_float(row.get("dynamic_ev_target"))
            for _, row in selected.iterrows()
        }
        for idx in batch_rows["_decision_idx"].astype(int).tolist():
            decision = decisions[idx]
            selected_flag = idx in selected_scores
            score = selected_scores.get(idx, 0.0)
            chain = dict(decision.get("chain_results") or {})
            payload = {
                "threshold_basis_policy_id": policy_id,
                "threshold_basis_family": family,
                "threshold_basis_window_days": int(window_days),
                "threshold_basis_rank_score": float(np.clip(score, 0.0, 1.0)),
                "threshold_basis_rank_score_source": f"threshold_basis:{policy_id}",
                "threshold_basis_selected": bool(selected_flag),
                "threshold_basis_reason": "selected" if selected_flag else "not_selected",
                "threshold_basis_dynamic_ev_target": selected_targets.get(idx, np.nan),
                "threshold_basis_dynamic_score_threshold": selected_thresholds.get(idx, np.nan),
                "threshold_basis_recent_reference_rows": int(meta["recent_reference_rows"]),
                "threshold_basis_reference_rows": int(meta["reference_rows"]),
                "threshold_basis_baseline_activity_count": int(meta["baseline_activity_count"]),
                "threshold_basis_global_dynamic_ev_target": meta["global_dynamic_ev_target"],
                "threshold_basis_global_dynamic_score_threshold": meta[
                    "global_dynamic_score_threshold"
                ],
            }
            decision.update(payload)
            chain.update(payload)
            decision["chain_results"] = chain
    return decisions
