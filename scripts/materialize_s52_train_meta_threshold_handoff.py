#!/usr/bin/env python3
"""Materialize an S52 train-meta threshold smoke into a replay/audit handoff.

This script freezes one diagnostic policy from
``run_s52_train_meta_regime_handoff_smoke.py`` into:

* a clean decision-time candidate file without realized outcomes;
* an offline evaluation candidate file with realized labels for audit only;
* a compact execution-plan CSV with side/source/policy parameters.

It does not refit models or reselect thresholds.  It applies a fixed policy
template to already materialized month-forward OOS predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1"
)
DEFAULT_SMOKE_DIR = DEFAULT_HANDOFF_DIR / "train_meta_regime_handoff_smoke_v1"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "s52_meta_threshold_top10_longaware_sidebad055_v1"
DEFAULT_SELECTOR = "meta_long_aware_clean_minus_risk"
DEFAULT_POLICY_ID = "side_bad_path_le_0.55"
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
OUTCOME_OR_LABEL_COLUMNS = {
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "u_policy_net",
    "ret_net",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_full_path_mae_norm",
    "underwater_bars_before_mfe_1r",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_drawdown_norm",
    "long_path_post_mfe_bad_drawdown",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_bad_path_label",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}")


def _score_col_for_selector(selector: str) -> str:
    return f"score_{selector}" if not selector.startswith("score_") else selector


def _side_key(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        side = frame["side_name"].astype(str).str.lower()
    elif "side" in frame.columns:
        side_num = pd.to_numeric(frame["side"], errors="coerce")
        side = pd.Series(np.where(side_num < 0.0, "short", "long"), index=frame.index)
    else:
        return pd.Series("all", index=frame.index)
    return pd.Series(
        np.select(
            [side.str.startswith("short"), side.str.startswith("long")],
            ["short", "long"],
            default="all",
        ),
        index=frame.index,
        dtype="object",
    )


def _find_score_column(frame: pd.DataFrame, preferred: str | None, selector_score_col: str) -> str:
    candidates = [
        preferred,
        selector_score_col,
        "meta_threshold_raw_score",
        "meta_threshold_score",
        "meta_score_oof",
        "exec_guard_score_oof",
        "calibrated_score",
        "score",
    ]
    for col in candidates:
        if col and col in frame.columns:
            return str(col)
    raise ValueError(
        "Could not find a rank-reference score column. "
        f"Tried {[col for col in candidates if col]} in {sorted(frame.columns)[:30]}..."
    )


def _rank_pct_against_reference(scores: pd.Series, reference_scores: pd.Series) -> tuple[pd.Series, int]:
    ref = pd.to_numeric(reference_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ref_arr = np.sort(ref.to_numpy(dtype=np.float64, copy=False))
    if ref_arr.size == 0:
        return pd.Series(np.nan, index=scores.index, dtype="float64"), 0
    score_arr = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ranks = np.full(score_arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(score_arr)
    ranks[finite] = np.searchsorted(ref_arr, score_arr[finite], side="right") / float(ref_arr.size)
    return pd.Series(ranks, index=scores.index, dtype="float64").clip(0.0, 1.0), int(ref_arr.size)


def _historical_rank_reference(
    predictions: pd.DataFrame,
    *,
    score_col: str,
    rank_reference_path: Path | None,
    rank_reference_score_col: str | None,
    rank_reference_scope: str,
    min_rank_reference_rows: int,
    allow_insample_rank_reference: bool,
) -> pd.DataFrame:
    """Map raw scores to causal historical percentiles.

    If a frozen reference table is supplied, it is treated as already historical
    relative to the scored rows.  Otherwise each month is ranked only against
    strictly prior rows present in the predictions table.  In-sample fallback is
    explicit and off by default because it is not inference-equivalent.
    """

    scope = str(rank_reference_scope or "global").lower()
    if scope not in {"global", "side"}:
        raise ValueError(f"Unsupported rank reference scope {rank_reference_scope!r}")
    work = predictions.copy()
    scores = _num(work[score_col], index=work.index)
    out = pd.DataFrame(index=work.index)
    out["meta_threshold_raw_score"] = scores.astype(np.float32)
    out["meta_threshold_rank_pct"] = np.nan
    out["rank_reference_n"] = 0
    out["rank_reference_scope"] = scope
    out["rank_reference_source"] = "none"
    out["rank_reference_score_col"] = score_col
    out["rank_reference_is_in_sample"] = False

    if rank_reference_path is not None:
        reference = _read_table(rank_reference_path)
        ref_score_col = _find_score_column(reference, rank_reference_score_col, score_col)
        ref_scores = _num(reference[ref_score_col])
        out["rank_reference_source"] = str(rank_reference_path)
        out["rank_reference_score_col"] = ref_score_col
        if scope == "side":
            row_side = _side_key(work)
            ref_side = _side_key(reference)
            for side_name in sorted(row_side.dropna().unique()):
                row_mask = row_side.eq(side_name)
                ref_mask = ref_side.eq(side_name)
                ranks, n_ref = _rank_pct_against_reference(scores.loc[row_mask], ref_scores.loc[ref_mask])
                if n_ref < int(min_rank_reference_rows):
                    ranks, n_ref = _rank_pct_against_reference(scores.loc[row_mask], ref_scores)
                    out.loc[row_mask, "rank_reference_scope"] = "global_fallback_from_side"
                out.loc[row_mask, "meta_threshold_rank_pct"] = ranks
                out.loc[row_mask, "rank_reference_n"] = int(n_ref)
        else:
            ranks, n_ref = _rank_pct_against_reference(scores, ref_scores)
            out["meta_threshold_rank_pct"] = ranks
            out["rank_reference_n"] = int(n_ref)
        return out

    if "month" in work.columns:
        month_values = work["month"].astype(str)
    else:
        ts = pd.to_datetime(work.get("__ts__", work.get("timestamp")), utc=True, errors="coerce")
        month_values = ts.dt.strftime("%Y-%m").astype(str)
    ts_values = pd.to_datetime(work.get("__ts__", work.get("timestamp")), utc=True, errors="coerce")
    side_values = _side_key(work)
    month_order = (
        pd.DataFrame({"month": month_values, "ts": ts_values})
        .dropna(subset=["ts"])
        .groupby("month")["ts"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    for month in month_order:
        row_mask = month_values.eq(str(month))
        month_start = ts_values.loc[row_mask].min()
        prior_mask = ts_values.notna() & ts_values.lt(month_start)
        out.loc[row_mask, "rank_reference_source"] = "prior_prediction_months"
        if scope == "side":
            for side_name in sorted(side_values.loc[row_mask].dropna().unique()):
                sub_mask = row_mask & side_values.eq(side_name)
                ref_mask = prior_mask & side_values.eq(side_name)
                ranks, n_ref = _rank_pct_against_reference(scores.loc[sub_mask], scores.loc[ref_mask])
                if n_ref < int(min_rank_reference_rows):
                    ranks, n_ref = _rank_pct_against_reference(scores.loc[sub_mask], scores.loc[prior_mask])
                    out.loc[sub_mask, "rank_reference_scope"] = "global_fallback_from_side"
                out.loc[sub_mask, "meta_threshold_rank_pct"] = ranks
                out.loc[sub_mask, "rank_reference_n"] = int(n_ref)
        else:
            ranks, n_ref = _rank_pct_against_reference(scores.loc[row_mask], scores.loc[prior_mask])
            out.loc[row_mask, "meta_threshold_rank_pct"] = ranks
            out.loc[row_mask, "rank_reference_n"] = int(n_ref)

    missing_ref = out["rank_reference_n"].lt(int(min_rank_reference_rows)) | out["meta_threshold_rank_pct"].isna()
    if missing_ref.any() and allow_insample_rank_reference:
        if scope == "side":
            for side_name in sorted(side_values.loc[missing_ref].dropna().unique()):
                sub_mask = missing_ref & side_values.eq(side_name)
                ref_mask = side_values.eq(side_name)
                ranks, n_ref = _rank_pct_against_reference(scores.loc[sub_mask], scores.loc[ref_mask])
                out.loc[sub_mask, "meta_threshold_rank_pct"] = ranks
                out.loc[sub_mask, "rank_reference_n"] = int(n_ref)
        else:
            ranks, n_ref = _rank_pct_against_reference(scores.loc[missing_ref], scores)
            out.loc[missing_ref, "meta_threshold_rank_pct"] = ranks
            out.loc[missing_ref, "rank_reference_n"] = int(n_ref)
        out.loc[missing_ref, "rank_reference_source"] = "INSAMPLE_FALLBACK_NOT_FOR_PRODUCTION"
        out.loc[missing_ref, "rank_reference_is_in_sample"] = True
    return out


def _select_policy_rows(
    predictions: pd.DataFrame,
    *,
    selector: str,
    policy_id: str,
    budget_frac: float,
    max_side_share: float | None = None,
    selection_mode: str = "historical_rank",
    rank_threshold: float | None = None,
    rank_reference_path: Path | None = None,
    rank_reference_score_col: str | None = None,
    rank_reference_scope: str = "side",
    min_rank_reference_rows: int = 100,
    allow_insample_rank_reference: bool = False,
) -> pd.DataFrame:
    score_col = _score_col_for_selector(selector)
    if score_col not in predictions.columns:
        raise ValueError(f"Missing selector score column {score_col!r}")
    required_policy_cols: list[str] = []
    if "bad_path" in str(policy_id):
        required_policy_cols.append("score_meta_bad_path")
    if "timeout" in str(policy_id):
        required_policy_cols.append("score_meta_timeout")
    if "clean_ge" in str(policy_id):
        required_policy_cols.append("score_meta_clean_exec")
    if "positive_margin" in str(policy_id):
        required_policy_cols.append("score_meta_positive_margin")
    missing_policy_cols = [col for col in sorted(set(required_policy_cols)) if col not in predictions.columns]
    if missing_policy_cols:
        raise ValueError(
            f"Policy {policy_id!r} requires missing prediction columns: {missing_policy_cols}"
        )
    clean = _num(predictions.get("score_meta_clean_exec"), index=predictions.index).fillna(0.0)
    long_clean = _num(predictions.get("score_meta_long_clean_exec"), index=predictions.index)
    side = predictions.get("side_name", pd.Series("", index=predictions.index)).astype(str).str.lower()
    side_clean = clean.where(~side.eq("long"), long_clean.fillna(clean))
    positive_margin = _num(predictions.get("score_meta_positive_margin"), index=predictions.index).fillna(0.0)
    bad_path = _num(predictions.get("score_meta_bad_path"), index=predictions.index).fillna(1.0)
    long_bad_path = _num(predictions.get("score_meta_long_bad_path"), index=predictions.index)
    side_bad_path = bad_path.where(~side.eq("long"), long_bad_path.fillna(bad_path))
    timeout = _num(predictions.get("score_meta_timeout"), index=predictions.index).fillna(1.0)

    eligible = pd.Series(True, index=predictions.index)
    for part in str(policy_id).split("_"):
        # Parsed by explicit cases below; this loop is intentionally inert but
        # keeps policy ids human-readable without ad-hoc regex side effects.
        _ = part
    if policy_id.startswith("clean_ge_"):
        eligible &= clean.ge(float(policy_id.rsplit("_", 1)[-1]))
    elif policy_id.startswith("side_clean_ge_long_") and "_short_" in policy_id:
        left, right = policy_id.replace("side_clean_ge_long_", "", 1).split("_short_", 1)
        long_floor = float(left)
        short_floor = float(right)
        eligible &= np.where(side.eq("long"), side_clean.ge(long_floor), clean.ge(short_floor))
    elif policy_id.startswith("side_clean_ge_"):
        eligible &= side_clean.ge(float(policy_id.rsplit("_", 1)[-1]))
    elif policy_id.startswith("side_bad_path_le_") and "_side_clean_ge_" in policy_id:
        left, right = policy_id.split("_side_clean_ge_", 1)
        eligible &= side_bad_path.le(float(left.rsplit("_", 1)[-1]))
        eligible &= side_clean.ge(float(right))
    elif policy_id.startswith("side_bad_path_le_"):
        eligible &= side_bad_path.le(float(policy_id.rsplit("_", 1)[-1]))
    elif policy_id.startswith("positive_margin_ge_"):
        eligible &= positive_margin.ge(float(policy_id.rsplit("_", 1)[-1]))
    elif policy_id.startswith("bad_path_le_") and "_clean_ge_" in policy_id:
        left, right = policy_id.split("_clean_ge_", 1)
        eligible &= bad_path.le(float(left.rsplit("_", 1)[-1]))
        eligible &= clean.ge(float(right))
    elif policy_id.startswith("bad_path_le_") and "_timeout_le_" in policy_id:
        left, right = policy_id.split("_timeout_le_", 1)
        eligible &= bad_path.le(float(left.rsplit("_", 1)[-1]))
        eligible &= timeout.le(float(right))
    elif policy_id.startswith("bad_path_le_"):
        eligible &= bad_path.le(float(policy_id.rsplit("_", 1)[-1]))
    elif policy_id != "no_cap":
        raise ValueError(f"Unsupported policy id {policy_id!r}")

    selection_mode = str(selection_mode or "historical_rank")
    threshold = float(rank_threshold) if rank_threshold is not None else 1.0 - float(budget_frac)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    work = predictions.copy()
    rank_info = _historical_rank_reference(
        work,
        score_col=score_col,
        rank_reference_path=rank_reference_path,
        rank_reference_score_col=rank_reference_score_col,
        rank_reference_scope=rank_reference_scope,
        min_rank_reference_rows=int(min_rank_reference_rows),
        allow_insample_rank_reference=bool(allow_insample_rank_reference),
    )
    for col in rank_info.columns:
        work[col] = rank_info[col].to_numpy()

    if selection_mode == "historical_rank":
        if max_side_share is not None and float(max_side_share) < 1.0:
            raise ValueError(
                "max_side_share is only supported by legacy month_budget mode. "
                "Historical-rank selection must use causal rank thresholds; side/portfolio caps belong downstream."
            )
        rank_pct = _num(work["meta_threshold_rank_pct"], index=work.index)
        valid_ref = _num(work["rank_reference_n"], index=work.index).ge(float(min_rank_reference_rows))
        selected_mask = eligible & _num(work[score_col], index=work.index).notna() & valid_ref & rank_pct.ge(threshold)
        out = work.loc[selected_mask].copy()
        if "__ts__" in out.columns:
            out = out.sort_values(["__ts__", "meta_threshold_rank_pct", score_col], ascending=[True, False, False], kind="mergesort")
        else:
            out = out.sort_values(["meta_threshold_rank_pct", score_col], ascending=[False, False], kind="mergesort")
        month_col = work["month"] if "month" in work.columns else pd.Series("unknown", index=work.index)
        expected = month_col.groupby(month_col).transform(lambda s: max(1, int(math.ceil(len(s) * float(budget_frac)))))
        selected_counts = pd.Series(False, index=work.index)
        selected_counts.loc[out.index] = True
        month_selected = selected_counts.groupby(month_col).transform("sum")
        work["target_rows_for_month"] = expected.astype(int)
        work["selected_rows_for_month"] = month_selected.astype(int)
        work["fill_rate_for_month"] = month_selected.astype(float) / expected.astype(float).replace(0.0, np.nan)
        for col in ("target_rows_for_month", "selected_rows_for_month", "fill_rate_for_month"):
            out[col] = work.loc[out.index, col].to_numpy()
        out["max_side_share_cap"] = float(max_side_share) if max_side_share is not None else 1.0
    elif selection_mode == "month_budget":
        selected_frames: list[pd.DataFrame] = []
        for month, group in work.groupby("month", dropna=False):
            target_rows = max(1, int(math.ceil(len(group) * float(budget_frac))))
            month_eligible = group[eligible.reindex(group.index).fillna(False)]
            ranked = month_eligible[_num(month_eligible[score_col]).notna()].sort_values(
                score_col,
                ascending=False,
                kind="mergesort",
            )
            if max_side_share is not None and float(max_side_share) < 1.0:
                max_per_side = max(1, int(math.floor(float(max_side_share) * target_rows)))
                counts = {"long": 0, "short": 0}
                selected_idx: list[int] = []
                for idx, row in ranked.iterrows():
                    side_key = "short" if str(row.get("side_name", "")).lower() == "short" else "long"
                    if counts[side_key] >= max_per_side:
                        continue
                    selected_idx.append(idx)
                    counts[side_key] += 1
                    if len(selected_idx) >= target_rows:
                        break
                chosen = ranked.loc[selected_idx].copy()
                while len(chosen) > 1:
                    side_counts = chosen["side_name"].astype(str).str.lower().value_counts()
                    dominant_side = str(side_counts.index[0]) if len(side_counts) else ""
                    dominant_share = float(side_counts.iloc[0] / max(len(chosen), 1)) if len(side_counts) else 0.0
                    if dominant_share <= float(max_side_share):
                        break
                    drop_candidates = chosen[
                        chosen["side_name"].astype(str).str.lower().eq(dominant_side)
                    ]
                    if drop_candidates.empty:
                        break
                    chosen = chosen.drop(index=drop_candidates.index[-1])
            else:
                chosen = ranked.head(target_rows).copy()
            chosen = chosen.copy()
            chosen["target_rows_for_month"] = int(target_rows)
            chosen["selected_rows_for_month"] = int(len(chosen))
            chosen["fill_rate_for_month"] = float(len(chosen) / max(target_rows, 1))
            chosen["max_side_share_cap"] = float(max_side_share) if max_side_share is not None else 1.0
            selected_frames.append(chosen)
        out = pd.concat(selected_frames, ignore_index=False) if selected_frames else work.iloc[0:0].copy()
    else:
        raise ValueError(f"Unsupported selection mode {selection_mode!r}")
    out = out.copy().reset_index(drop=True)
    out["meta_threshold_selector"] = selector
    out["meta_threshold_policy_id"] = policy_id
    out["meta_threshold_budget_frac"] = float(budget_frac)
    out["meta_threshold_rank_threshold"] = np.float32(threshold)
    out["meta_threshold_selection_mode"] = selection_mode
    out["meta_threshold_score"] = _num(out[score_col]).astype(np.float32)
    out["meta_threshold_raw_score"] = _num(out.get("meta_threshold_raw_score", out[score_col]), index=out.index).astype(np.float32)
    out["meta_threshold_rank_pct"] = _num(out.get("meta_threshold_rank_pct"), index=out.index).astype(np.float32)
    out["policy_rank_pct"] = out["meta_threshold_rank_pct"]
    out["historical_rank_pct"] = out["meta_threshold_rank_pct"]
    out["threshold_rank_score"] = out["meta_threshold_rank_pct"]
    out["rank_score_source"] = "historical_score_reference_percentile" if selection_mode == "historical_rank" else "legacy_month_budget_score_rank"
    out["meta_clean_exec_score_oos"] = _num(out.get("score_meta_clean_exec"), index=out.index).astype(np.float32)
    if "score_meta_long_clean_exec" in out.columns:
        out["meta_long_clean_exec_score_oos"] = _num(out.get("score_meta_long_clean_exec"), index=out.index).astype(
            np.float32
        )
    if "score_meta_long_bad_path" in out.columns:
        out["meta_long_bad_path_score_oos"] = _num(out.get("score_meta_long_bad_path"), index=out.index).astype(
            np.float32
        )
    out["meta_positive_margin_score_oos"] = _num(out.get("score_meta_positive_margin"), index=out.index).astype(np.float32)
    for src, dst in (
        ("score_meta_mfe_before_mae", "meta_mfe_before_mae_score_oos"),
        ("score_meta_mae_before_mfe", "meta_mae_before_mfe_score_oos"),
        ("score_meta_underwater_duration", "meta_underwater_duration_score_oos"),
        ("score_meta_path_order", "meta_path_order_score_oos"),
        ("score_meta_path_order_clean_minus_risk", "meta_path_order_clean_minus_risk_score_oos"),
    ):
        if src in out.columns:
            out[dst] = _num(out.get(src), index=out.index).astype(np.float32)
    out_side = out["side_name"].astype(str).str.lower()
    out_clean = _num(out.get("score_meta_clean_exec"), index=out.index).fillna(0.0)
    out_long_clean = _num(out.get("score_meta_long_clean_exec"), index=out.index)
    out_bad = _num(out.get("score_meta_bad_path"), index=out.index).fillna(1.0)
    out_long_bad = _num(out.get("score_meta_long_bad_path"), index=out.index)
    out["meta_side_clean_exec_score_oos"] = out_clean.where(~out_side.eq("long"), out_long_clean.fillna(out_clean)).astype(
        np.float32
    )
    out["meta_side_bad_path_score_oos"] = out_bad.where(~out_side.eq("long"), out_long_bad.fillna(out_bad)).astype(
        np.float32
    )
    if policy_id.startswith("side_clean_ge_long_") and "_short_" in policy_id:
        left, right = policy_id.replace("side_clean_ge_long_", "", 1).split("_short_", 1)
        out["meta_side_clean_threshold"] = np.where(
            out["side_name"].astype(str).str.lower().eq("long"),
            float(left),
            float(right),
        ).astype(np.float32)
    elif policy_id.startswith("side_clean_ge_"):
        out["meta_side_clean_threshold"] = np.float32(float(policy_id.rsplit("_", 1)[-1]))
    return out


def _join_handoff_context(selected: pd.DataFrame, handoff_path: Path) -> pd.DataFrame:
    handoff = pd.read_parquet(handoff_path)
    context_cols = [
        "__ts__",
        "__symbol__",
        "side_name",
        "score",
        "source_tag",
        "source_family",
        "source_semantic_family",
        "source_semantic_family_base",
        "long_source_regime_split",
        "source_volatility_state",
        "source_pressure_state",
        "source_trend_state",
        "source_recipe_tag",
        "__archetype_label_family__",
        "__archetype_label_source__",
        "__archetype_policy_key__",
        "__archetype_policy_role__",
        "__archetype_policy_confidence__",
        "__archetype_policy_tp_r__",
        "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
        "__archetype_policy_max_bars_to_mfe__",
        "__archetype_policy_max_barrier__",
        "archetype_label_family",
        "archetype_label_source",
        "archetype_policy_key",
        "archetype_policy_role",
        "archetype_policy_confidence",
        "archetype_policy_tp_r",
        "archetype_policy_sl_r",
        "archetype_policy_trail_r",
        "archetype_policy_max_bars_to_mfe",
        "archetype_policy_max_barrier",
        "base_score_decile",
        "base_rank_band",
        "base_margin_band",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_signal_zscore_within_archetype",
        "base_score_rank_pct_train_prior",
        "aegmm_cluster",
        "aegmm_entropy_bin",
        "aegmm_distance_bin",
        "aegmm_expected_distance_bin",
        "reconstruction_bin",
        "dae_reconstruction_bin",
        "latent_speed_bin",
        "side_aegmm_cluster",
        "regime_first_touch_bad_mae_score_bin",
        "regime_timeout_score_bin",
        "regime_dirty_positive_score_bin",
        "regime_clean_exec_score_bin",
        "regime_lgbm_leaf_bad_mae_k4",
        "regime_lgbm_leaf_exec_margin_k4",
        "gmm_cluster_id",
        "gmm_entropy",
        "mahalanobis_distance",
        "AE_reconstruction_error",
        "dae_reconstruction_error",
        "latent_speed",
        "latent_acceleration",
        "meta_context_weight_hint",
        "meta_threshold_adjustment_hint",
    ]
    payload = handoff[[col for col in context_cols if col in handoff.columns]].copy()
    return selected.merge(payload, on=list(KEY_COLUMNS), how="left", validate="one_to_one", suffixes=("", "__handoff"))


def _side_to_numeric(side_name: pd.Series) -> pd.Series:
    return pd.Series(np.where(side_name.astype(str).str.lower().eq("short"), -1.0, 1.0), index=side_name.index)


def _clean_handoff(frame: pd.DataFrame, *, source_hash: str, policy_hash: str) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = out["__ts__"]
    out["symbol"] = out["__symbol__"]
    out["side"] = _side_to_numeric(out["side_name"])
    out["signal_timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["guard_decision_timestamp"] = out["signal_timestamp"]
    out["guard_decision_contract"] = "signal_time_oos_meta_threshold"
    out["requires_delayed_entry_observation"] = False
    out["accepted"] = True
    out["guard_accepted"] = True
    out["base_selector"] = "s52_trailing_top10_pointwise_lgbm"
    out["base_score_oof"] = _num(out.get("score_base"), index=out.index)
    out["meta_score_oof"] = _num(out.get("meta_threshold_score"), index=out.index)
    out["exec_guard_score_oof"] = out["meta_score_oof"]
    out["exec_guard_threshold"] = _num(out.get("meta_side_clean_threshold"), index=out.index)
    out["exec_guard_method"] = out["meta_threshold_policy_id"]
    out["threshold_source"] = np.where(
        out.get("meta_threshold_selection_mode", pd.Series("", index=out.index)).astype(str).eq("historical_rank"),
        "historical_score_reference_top_fraction_not_validation_optimized",
        "legacy_month_budget_not_inference_equivalent",
    )
    out["decision_fold"] = out["month"].astype(str).map({month: idx for idx, month in enumerate(sorted(out["month"].astype(str).unique()), start=1)})
    out["train_window_end"] = np.where(out["month"].astype(str).eq("2026-05"), "2026-04", "2026-05")
    out["scenario_id"] = "S52_trailing_tp075_sl050_tr035_fast12_bar30"
    out["scenario_family"] = "fixed_trailing_profit"
    out["horizon_bars"] = 30
    out["horizon_hours"] = 7.5
    out["tp_activation_r"] = 0.75
    out["stop_mult"] = 0.50
    out["trail_gap_r"] = 0.35
    out["max_activation_bars"] = 12
    out["round_trip_cost_floor"] = 0.01
    out["feature_set_hash"] = source_hash
    out["model_hash"] = policy_hash
    out["handoff_row_id"] = [f"s52_meta_threshold_{idx:06d}" for idx in range(len(out))]
    keep = [
        "handoff_row_id",
        "timestamp",
        "symbol",
        "side_name",
        "side",
        "signal_timestamp",
        "guard_decision_timestamp",
        "guard_decision_contract",
        "requires_delayed_entry_observation",
        "month",
        "source_semantic_family",
        "source_semantic_family_base",
        "long_source_regime_split",
        "source_tag",
        "source_family",
        "source_volatility_state",
        "source_pressure_state",
        "source_trend_state",
        "source_recipe_tag",
        "__archetype_label_family__",
        "__archetype_label_source__",
        "__archetype_policy_key__",
        "__archetype_policy_role__",
        "__archetype_policy_confidence__",
        "__archetype_policy_tp_r__",
        "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
        "__archetype_policy_max_bars_to_mfe__",
        "__archetype_policy_max_barrier__",
        "archetype_label_family",
        "archetype_label_source",
        "archetype_policy_key",
        "archetype_policy_role",
        "archetype_policy_confidence",
        "archetype_policy_tp_r",
        "archetype_policy_sl_r",
        "archetype_policy_trail_r",
        "archetype_policy_max_bars_to_mfe",
        "archetype_policy_max_barrier",
        "base_selector",
        "base_score_oof",
        "meta_score_oof",
        "meta_clean_exec_score_oos",
        "meta_long_clean_exec_score_oos",
        "meta_long_bad_path_score_oos",
        "meta_side_clean_exec_score_oos",
        "meta_side_bad_path_score_oos",
        "meta_side_clean_threshold",
        "meta_positive_margin_score_oos",
        "meta_mfe_before_mae_score_oos",
        "meta_mae_before_mfe_score_oos",
        "meta_underwater_duration_score_oos",
        "meta_path_order_score_oos",
        "meta_path_order_clean_minus_risk_score_oos",
        "meta_threshold_selector",
        "meta_threshold_policy_id",
        "meta_threshold_budget_frac",
        "meta_threshold_selection_mode",
        "meta_threshold_rank_threshold",
        "meta_threshold_raw_score",
        "meta_threshold_rank_pct",
        "policy_rank_pct",
        "historical_rank_pct",
        "threshold_rank_score",
        "rank_score_source",
        "rank_reference_source",
        "rank_reference_score_col",
        "rank_reference_scope",
        "rank_reference_n",
        "rank_reference_is_in_sample",
        "max_side_share_cap",
        "target_rows_for_month",
        "selected_rows_for_month",
        "fill_rate_for_month",
        "scenario_id",
        "scenario_family",
        "horizon_bars",
        "horizon_hours",
        "tp_activation_r",
        "stop_mult",
        "trail_gap_r",
        "max_activation_bars",
        "round_trip_cost_floor",
        "accepted",
        "guard_accepted",
        "exec_guard_score_oof",
        "exec_guard_threshold",
        "exec_guard_method",
        "threshold_source",
        "decision_fold",
        "train_window_end",
        "base_score_decile",
        "base_rank_band",
        "base_margin_band",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_signal_zscore_within_archetype",
        "base_score_rank_pct_train_prior",
        "aegmm_cluster",
        "aegmm_entropy_bin",
        "aegmm_distance_bin",
        "aegmm_expected_distance_bin",
        "reconstruction_bin",
        "dae_reconstruction_bin",
        "latent_speed_bin",
        "side_aegmm_cluster",
        "regime_first_touch_bad_mae_score_bin",
        "regime_timeout_score_bin",
        "regime_dirty_positive_score_bin",
        "regime_clean_exec_score_bin",
        "regime_lgbm_leaf_bad_mae_k4",
        "regime_lgbm_leaf_exec_margin_k4",
        "gmm_cluster_id",
        "gmm_entropy",
        "mahalanobis_distance",
        "AE_reconstruction_error",
        "dae_reconstruction_error",
        "latent_speed",
        "latent_acceleration",
        "meta_context_weight_hint",
        "meta_threshold_adjustment_hint",
        "feature_set_hash",
        "model_hash",
    ]
    keep = [col for col in keep if col in out.columns and col not in OUTCOME_OR_LABEL_COLUMNS]
    return out[keep].copy()


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col in OUTCOME_OR_LABEL_COLUMNS]


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    month_exec = (
        frame.groupby("month")["exec_margin"].mean()
        if {"month", "exec_margin"}.issubset(frame.columns) and not frame.empty
        else pd.Series(dtype=float)
    )
    return {
        "rows": int(len(frame)),
        "months": sorted(str(m) for m in frame.get("month", pd.Series(dtype=str)).dropna().unique()),
        "symbols": int(frame.get("__symbol__", pd.Series(dtype=str)).nunique()),
        "mean_exec_margin": _mean(frame.get("exec_margin")),
        "worst_month_exec_margin": float(month_exec.min()) if len(month_exec) else float("nan"),
        "mean_ret_net": _mean(frame.get("ret_net")),
        "mean_u_policy_net": _mean(frame.get("u_policy_net")),
        "full_path_bad_mae": _rate(frame.get("full_path_bad_mae_1r")),
        "max_month_full_path_bad_mae": float(frame.groupby("month")["full_path_bad_mae_1r"].mean().max())
        if {"month", "full_path_bad_mae_1r"}.issubset(frame.columns)
        else float("nan"),
        "timeout": _rate(frame.get("timeout")),
        "clean_exec_precision": _rate(frame.get("clean_exec")),
        "positive_margin_rate": _rate(_num(frame.get("exec_margin")).gt(0.0)),
        "short_share": float(frame["side_name"].astype(str).eq("short").mean()) if "side_name" in frame.columns and len(frame) else float("nan"),
    }


def _month_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for month, group in frame.groupby("month", dropna=False):
        rows.append(
            {
                "month": month,
                "rows": int(len(group)),
                "symbols": int(group["__symbol__"].nunique()) if "__symbol__" in group else 0,
                "exec_margin": _mean(group.get("exec_margin")),
                "ret_net": _mean(group.get("ret_net")),
                "u_policy_net": _mean(group.get("u_policy_net")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "positive_margin_rate": _rate(_num(group.get("exec_margin")).gt(0.0)),
                "short_share": float(group["side_name"].astype(str).eq("short").mean()) if "side_name" in group else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _source_coverage_summary(predictions: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "month" not in predictions.columns:
        return pd.DataFrame()
    pred = predictions.copy()
    chosen_all = selected.copy()

    if "__ts__" in pred.columns:
        pred_ts = pd.to_datetime(pred["__ts__"], utc=True, errors="coerce")
    elif "timestamp" in pred.columns:
        pred_ts = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    else:
        pred_ts = pd.Series(pd.NaT, index=pred.index)
    if "__ts__" in chosen_all.columns:
        chosen_ts = pd.to_datetime(chosen_all["__ts__"], utc=True, errors="coerce")
    elif "timestamp" in chosen_all.columns:
        chosen_ts = pd.to_datetime(chosen_all["timestamp"], utc=True, errors="coerce")
    else:
        chosen_ts = pd.Series(pd.NaT, index=chosen_all.index)

    pred["_day"] = pred_ts.dt.date
    chosen_all["_day"] = chosen_ts.dt.date
    chosen_groups = (
        chosen_all.groupby("month", dropna=False) if "month" in chosen_all.columns else None
    )
    rows: list[dict[str, Any]] = []
    for month, group in pred.groupby("month", dropna=False):
        if chosen_groups is not None and month in chosen_groups.groups:
            chosen = chosen_groups.get_group(month)
        else:
            chosen = pd.DataFrame(columns=chosen_all.columns)
        target_rows = 0
        if not chosen.empty and "target_rows_for_month" in chosen.columns:
            target_values = pd.to_numeric(
                chosen["target_rows_for_month"],
                errors="coerce",
            ).dropna()
            if len(target_values):
                target_rows = int(target_values.iloc[0])
        pred_days = int(group["_day"].dropna().nunique())
        chosen_days = int(chosen["_day"].dropna().nunique()) if not chosen.empty else 0
        rows.append(
            {
                "month": month,
                "source_rows": int(len(group)),
                "selected_rows": int(len(chosen)),
                "target_rows": int(target_rows),
                "fill_rate": float(len(chosen) / max(target_rows, 1)) if target_rows else float("nan"),
                "active_source_days": pred_days,
                "active_selected_days": chosen_days,
                "source_symbols": int(group["__symbol__"].astype(str).nunique())
                if "__symbol__" in group
                else 0,
                "selected_symbols": int(chosen["__symbol__"].astype(str).nunique())
                if "__symbol__" in chosen
                else 0,
                "source_rows_per_active_day": float(len(group) / max(pred_days, 1)),
                "selected_rows_per_active_day": float(len(chosen) / max(chosen_days, 1))
                if chosen_days
                else 0.0,
                "selected_fraction_of_source": float(len(chosen) / max(len(group), 1)),
            }
        )
    return pd.DataFrame(rows)


def _source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "source_semantic_family" not in frame.columns:
        return pd.DataFrame()
    rows = []
    group_sets = [
        ["month", "side_name", "source_semantic_family"],
    ]
    for regime_col in ("aegmm_cluster", "side_aegmm_cluster", "aegmm_expected_distance_bin", "reconstruction_bin"):
        if regime_col in frame.columns:
            group_sets.append(["month", "side_name", "source_semantic_family", regime_col])
    for group_cols in group_sets:
        for keys, group in frame.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {
                "grouping": "+".join(group_cols),
                "rows": int(len(group)),
                "exec_margin": _mean(group.get("exec_margin")),
                "ret_net": _mean(group.get("ret_net")),
                "u_policy_net": _mean(group.get("u_policy_net")),
                "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
                "timeout": _rate(group.get("timeout")),
                "clean_exec_precision": _rate(group.get("clean_exec")),
                "positive_margin_rate": _rate(_num(group.get("exec_margin")).gt(0.0)),
                "short_share": float(group["side_name"].astype(str).eq("short").mean()) if "side_name" in group else float("nan"),
            }
            for col, value in zip(group_cols, keys, strict=False):
                record[col] = value
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["month", "rows"], ascending=[True, False])


def materialize(
    *,
    smoke_dir: Path,
    handoff_dir: Path,
    out_dir: Path,
    selector: str,
    policy_id: str,
    budget_frac: float,
    max_side_share: float | None = None,
    selection_mode: str = "historical_rank",
    rank_threshold: float | None = None,
    rank_reference_path: Path | None = None,
    rank_reference_score_col: str | None = None,
    rank_reference_scope: str = "side",
    min_rank_reference_rows: int = 100,
    allow_insample_rank_reference: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = smoke_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet"
    threshold_summary_path = smoke_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv"
    handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    if not threshold_summary_path.exists():
        raise FileNotFoundError(threshold_summary_path)
    if not handoff_path.exists():
        raise FileNotFoundError(handoff_path)
    predictions = pd.read_parquet(predictions_path)
    selected = _select_policy_rows(
        predictions,
        selector=selector,
        policy_id=policy_id,
        budget_frac=budget_frac,
        max_side_share=max_side_share,
        selection_mode=selection_mode,
        rank_threshold=rank_threshold,
        rank_reference_path=rank_reference_path,
        rank_reference_score_col=rank_reference_score_col,
        rank_reference_scope=rank_reference_scope,
        min_rank_reference_rows=min_rank_reference_rows,
        allow_insample_rank_reference=allow_insample_rank_reference,
    )
    selected = _join_handoff_context(selected, handoff_path)
    policy_payload = {
        "selector": selector,
        "policy_id": policy_id,
        "budget_frac": float(budget_frac),
        "max_side_share": float(max_side_share) if max_side_share is not None else None,
        "selection_mode": selection_mode,
        "rank_threshold": float(rank_threshold) if rank_threshold is not None else float(1.0 - float(budget_frac)),
        "rank_reference_path": str(rank_reference_path) if rank_reference_path is not None else None,
        "rank_reference_score_col": rank_reference_score_col,
        "rank_reference_scope": rank_reference_scope,
        "min_rank_reference_rows": int(min_rank_reference_rows),
        "allow_insample_rank_reference": bool(allow_insample_rank_reference),
        "source_predictions_sha256": _sha256_path(predictions_path),
        "source_threshold_summary_sha256": _sha256_path(threshold_summary_path),
    }
    policy_hash = _hash_payload(policy_payload)
    source_hash = _hash_payload(
        {
            "handoff_path": str(handoff_path),
            "handoff_sha256": _sha256_path(handoff_path),
            "prediction_path": str(predictions_path),
            "prediction_sha256": _sha256_path(predictions_path),
        }
    )
    clean = _clean_handoff(selected, source_hash=source_hash, policy_hash=policy_hash)
    offline = selected.copy()
    offline["handoff_row_id"] = clean["handoff_row_id"].to_numpy()
    execution_cols = [
        "handoff_row_id",
        "timestamp",
        "symbol",
        "side_name",
        "side",
        "month",
        "source_semantic_family",
        "meta_score_oof",
        "meta_clean_exec_score_oos",
        "scenario_id",
        "horizon_bars",
        "tp_activation_r",
        "stop_mult",
        "trail_gap_r",
        "round_trip_cost_floor",
    ]
    execution_plan = clean[[col for col in execution_cols if col in clean.columns]].copy()
    summary = _summarize(offline)
    month_summary = _month_summary(offline)
    source_coverage = _source_coverage_summary(predictions, selected)
    source_summary = _source_summary(offline)
    clean_forbidden = _forbidden_columns(clean)
    duplicate_keys = ["timestamp", "symbol", "side_name", "scenario_id"]
    duplicate_count = int(clean.duplicated([col for col in duplicate_keys if col in clean.columns]).sum()) if not clean.empty else 0
    paths = {
        "clean_handoff": out_dir / "s52_meta_threshold_guarded_candidates.parquet",
        "offline_eval_candidates": out_dir / "s52_meta_threshold_guarded_offline_eval_candidates.parquet",
        "execution_plan": out_dir / "s52_meta_threshold_guarded_execution_plan.csv",
        "summary": out_dir / "s52_meta_threshold_guarded_summary.csv",
        "month_summary": out_dir / "s52_meta_threshold_guarded_month_summary.csv",
        "source_coverage": out_dir / "s52_meta_threshold_guarded_source_coverage.csv",
        "source_summary": out_dir / "s52_meta_threshold_guarded_source_summary.csv",
        "leakage_audit": out_dir / "s52_meta_threshold_guarded_leakage_audit.json",
        "manifest": out_dir / "s52_meta_threshold_guarded_manifest.json",
        "report": out_dir / "s52_meta_threshold_guarded_report.md",
    }
    clean.to_parquet(paths["clean_handoff"], index=False)
    offline.to_parquet(paths["offline_eval_candidates"], index=False)
    execution_plan.to_csv(paths["execution_plan"], index=False)
    pd.DataFrame([summary]).to_csv(paths["summary"], index=False)
    month_summary.to_csv(paths["month_summary"], index=False)
    source_coverage.to_csv(paths["source_coverage"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    audit = {
        "clean_handoff_forbidden_columns": clean_forbidden,
        "clean_handoff_has_no_realized_outcomes": not clean_forbidden,
        "duplicate_decision_key_rows": duplicate_count,
        "policy_source": "fixed_smoke_template_oos_predictions",
        "selection_mode": selection_mode,
        "rank_reference_is_in_sample_rows": int(
            pd.Series(selected.get("rank_reference_is_in_sample", pd.Series(False, index=selected.index)))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "rank_reference_min_n": int(
            pd.to_numeric(selected.get("rank_reference_n", pd.Series(dtype=float)), errors="coerce").min()
        )
        if "rank_reference_n" in selected.columns and len(selected)
        else 0,
        "threshold_is_validation_optimized": False,
        "max_side_share_cap": float(max_side_share) if max_side_share is not None else None,
        "months": summary["months"],
        "clean_rows": int(len(clean)),
        "offline_rows": int(len(offline)),
    }
    paths["leakage_audit"].write_text(json.dumps(_json_safe(audit), indent=2, sort_keys=True))
    manifest = {
        "generated_by": "materialize_s52_train_meta_threshold_handoff",
        "smoke_dir": str(smoke_dir),
        "handoff_dir": str(handoff_dir),
        "policy": policy_payload,
        "policy_hash": policy_hash,
        "source_hash": source_hash,
        "outputs": {key: str(path) for key, path in paths.items()},
        "summary": summary,
        "leakage_audit": audit,
        "status": "materialized_threshold_handoff_candidate"
        if not clean_forbidden and duplicate_count == 0
        else "materialized_with_audit_warnings",
        "promotion_note": "Diagnostic handoff only; exact barwise replay and production train_meta integration still required.",
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(paths["report"], manifest, month_summary, source_summary, source_coverage)
    return manifest


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    month_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    source_coverage: pd.DataFrame,
) -> None:
    summary = manifest.get("summary", {})
    lines = [
        "# S52 Meta Threshold Handoff",
        "",
        "## Scope",
        "",
        "Materialized diagnostic OOS threshold handoff from the S52 train-meta smoke.",
        "This is a clean handoff candidate for audit/replay preparation, not production train_meta or frozen replay evidence.",
        "",
        "## Policy",
        "",
        f"- selector: `{manifest['policy']['selector']}`",
        f"- policy id: `{manifest['policy']['policy_id']}`",
        f"- selection mode: `{manifest['policy'].get('selection_mode')}`",
        f"- target top fraction: `{float(manifest['policy']['budget_frac']) * 100:.1f}%`",
        f"- historical rank threshold: `{float(manifest['policy'].get('rank_threshold', 1.0 - float(manifest['policy']['budget_frac']))):.4f}`",
        f"- rank reference scope: `{manifest['policy'].get('rank_reference_scope')}`",
        f"- rank reference path: `{manifest['policy'].get('rank_reference_path')}`",
        f"- max side share cap: `{manifest['policy'].get('max_side_share')}`",
        f"- status: `{manifest['status']}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- rows: `{summary.get('rows')}`",
        f"- symbols: `{summary.get('symbols')}`",
        f"- mean executable margin: `{_fmt_pct(summary.get('mean_exec_margin'))}`",
        f"- mean net return: `{_fmt_pct(summary.get('mean_ret_net'))}`",
        f"- mean policy utility net: `{_fmt_pct(summary.get('mean_u_policy_net'))}`",
        f"- full-path bad-MAE: `{_fmt_pct(summary.get('full_path_bad_mae'))}`",
        f"- max-month full-path bad-MAE: `{_fmt_pct(summary.get('max_month_full_path_bad_mae'))}`",
        f"- timeout: `{_fmt_pct(summary.get('timeout'))}`",
        f"- clean precision: `{_fmt_pct(summary.get('clean_exec_precision'))}`",
        f"- short share: `{_fmt_pct(summary.get('short_share'))}`",
        "",
        "## Month Metrics",
        "",
        month_summary.to_markdown(index=False) if not month_summary.empty else "_No rows._",
        "",
        "## Source Coverage",
        "",
        source_coverage.to_markdown(index=False) if not source_coverage.empty else "_No rows._",
        "",
        "## Source Metrics",
        "",
        source_summary.head(30).to_markdown(index=False) if not source_summary.empty else "_No rows._",
        "",
        "## Audit",
        "",
        f"- clean handoff forbidden columns: `{manifest['leakage_audit']['clean_handoff_forbidden_columns']}`",
        f"- duplicate decision keys: `{manifest['leakage_audit']['duplicate_decision_key_rows']}`",
        "- exact barwise replay remains required before promotion.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--policy-id", default=DEFAULT_POLICY_ID)
    parser.add_argument("--budget-frac", type=float, default=0.10)
    parser.add_argument(
        "--selection-mode",
        choices=("historical_rank", "month_budget"),
        default="historical_rank",
        help="historical_rank maps raw meta scores to a frozen/prior percentile before thresholding; month_budget is legacy.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=float,
        default=None,
        help="Historical rank threshold. Defaults to 1 - budget_frac, e.g. 0.90 for top 10%%.",
    )
    parser.add_argument(
        "--rank-reference-path",
        type=Path,
        default=None,
        help="Optional frozen historical score table used to convert raw scores to percentile ranks.",
    )
    parser.add_argument(
        "--rank-reference-score-col",
        default=None,
        help="Score column inside --rank-reference-path. Defaults to selector score with sensible fallbacks.",
    )
    parser.add_argument(
        "--rank-reference-scope",
        choices=("global", "side"),
        default="side",
        help="Use a global historical score distribution or side-specific distributions.",
    )
    parser.add_argument("--min-rank-reference-rows", type=int, default=100)
    parser.add_argument(
        "--allow-insample-rank-reference",
        action="store_true",
        help="Diagnostic-only fallback when no prior/frozen rank reference exists. Do not use for production replay.",
    )
    parser.add_argument(
        "--max-side-share",
        type=float,
        default=1.0,
        help="Optional per-month cap on either side's selected share. Values >=1 disable the cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = materialize(
        smoke_dir=args.smoke_dir,
        handoff_dir=args.handoff_dir,
        out_dir=args.out_dir,
        selector=str(args.selector),
        policy_id=str(args.policy_id),
        budget_frac=float(args.budget_frac),
        max_side_share=float(args.max_side_share) if float(args.max_side_share) < 1.0 else None,
        selection_mode=str(args.selection_mode),
        rank_threshold=float(args.rank_threshold) if args.rank_threshold is not None else None,
        rank_reference_path=args.rank_reference_path,
        rank_reference_score_col=args.rank_reference_score_col,
        rank_reference_scope=str(args.rank_reference_scope),
        min_rank_reference_rows=int(args.min_rank_reference_rows),
        allow_insample_rank_reference=bool(args.allow_insample_rank_reference),
    )
    print(json.dumps(_json_safe({"event": "s52_meta_threshold_handoff_done", **manifest}), sort_keys=True))


if __name__ == "__main__":
    main()
