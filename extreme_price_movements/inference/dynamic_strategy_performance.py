"""Dynamic strategy-level performance calibration for live inference.

The monitor is deliberately outside the hot model path. It reads the persisted
prediction ledger, compares recent top predictions with the optimiser OOS
baseline, and returns a bounded strategy x meta-head threshold multiplier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.utils import tprint


DYNAMIC_PERFORMANCE_PATH = "live_state/dynamic_strategy_performance.json"
_META_HEAD_HASH_CACHE: dict[tuple[int, str, str], str] = {}


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True))
    tmp.replace(path)


def meta_head_hash(
    *,
    meta_model_key: str,
    meta_model: Any,
    feature_contract: Optional[list[str]] = None,
) -> str:
    """Return a stable short hash for a deployed meta head.

    The hash intentionally includes the model key and selected feature contract,
    so a retrained or reselected head uses an independent performance ledger even
    if the strategy id is unchanged. If the estimator exposes a booster string,
    include that too without serialising arbitrary Python objects.
    """

    contract_sig = hashlib.sha256(
        "\0".join(str(c) for c in (feature_contract or [])).encode(
            "utf-8", errors="ignore"
        )
    ).hexdigest()[:16]
    cache_key = (id(meta_model), str(meta_model_key or ""), contract_sig)
    cached = _META_HEAD_HASH_CACHE.get(cache_key)
    if cached:
        return cached
    h = hashlib.sha256()
    h.update(str(meta_model_key or "").encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(type(meta_model).__name__.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    if feature_contract:
        for col in feature_contract:
            h.update(str(col).encode("utf-8", errors="ignore"))
            h.update(b"\0")
    for attr in ("best_iteration_", "n_features_in_", "feature_name_"):
        try:
            h.update(repr(getattr(meta_model, attr, "")).encode("utf-8", errors="ignore"))
            h.update(b"\n")
        except Exception:
            pass
    booster = getattr(meta_model, "booster_", None)
    if booster is not None:
        try:
            h.update(str(booster.model_to_string(num_iteration=0)).encode("utf-8", errors="ignore"))
        except Exception:
            try:
                h.update(repr(booster).encode("utf-8", errors="ignore"))
            except Exception:
                pass
    out = h.hexdigest()[:16]
    _META_HEAD_HASH_CACHE[cache_key] = out
    return out


def _psi(recent: pd.Series, baseline: pd.Series, *, bins: int = 10) -> float:
    recent = pd.to_numeric(recent, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    baseline = pd.to_numeric(baseline, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(recent) < 10 or len(baseline) < 50:
        return float("nan")
    qs = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.unique(np.nanquantile(baseline.to_numpy(dtype=float), qs))
    if edges.size < 3:
        return float("nan")
    edges[0] = -np.inf
    edges[-1] = np.inf
    recent_counts = np.histogram(recent.to_numpy(dtype=float), bins=edges)[0].astype(float)
    base_counts = np.histogram(baseline.to_numpy(dtype=float), bins=edges)[0].astype(float)
    r = np.maximum(recent_counts / max(recent_counts.sum(), 1.0), 1e-6)
    b = np.maximum(base_counts / max(base_counts.sum(), 1.0), 1e-6)
    return float(np.sum((r - b) * np.log(r / b)))


def _jsd(recent: pd.Series, baseline: pd.Series) -> float:
    recent = recent.dropna().astype(str)
    baseline = baseline.dropna().astype(str)
    if len(recent) == 0 or len(baseline) == 0:
        return float("nan")
    keys = sorted(set(recent.unique()).union(set(baseline.unique())))
    r = recent.value_counts(normalize=True).reindex(keys, fill_value=0.0).to_numpy(dtype=float)
    b = baseline.value_counts(normalize=True).reindex(keys, fill_value=0.0).to_numpy(dtype=float)
    m = 0.5 * (r + b)
    mask_r = r > 0
    mask_b = b > 0
    kl_r = float(np.sum(r[mask_r] * np.log(r[mask_r] / m[mask_r])))
    kl_b = float(np.sum(b[mask_b] * np.log(b[mask_b] / m[mask_b])))
    return float(0.5 * (kl_r + kl_b))


def _time_decay_weights(ts: pd.Series, now: pd.Timestamp) -> np.ndarray:
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    age_days = (now - parsed).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    # Linear decay requested by user: 10 days old is 50%, 20+ days is 0%.
    return np.clip(1.0 - age_days / 20.0, 0.0, 1.0)


def _rank_weights(df: pd.DataFrame) -> np.ndarray:
    for col in ("policy_rank_pct", "threshold_rank_score", "normalized_rank_score", "meta_train_rank_pct"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, 1.0), 0.5)
            return 0.5 + 0.5 * vals
    return np.ones(len(df), dtype=float)


def _outcome_hit(df: pd.DataFrame) -> pd.Series:
    for col in ("net_pnl_pct", "realized_net_pnl_pct", "net_return"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            return (vals > 0.0).where(vals.notna())
    if "tp_hit" in df.columns:
        vals = df["tp_hit"]
        return vals.where(vals.notna())
    if "outcome_status" in df.columns:
        status = df["outcome_status"].astype(str).str.lower()
        return status.isin({"win", "tp_hit", "profit", "positive"}).where(
            status.isin({"win", "tp_hit", "profit", "positive", "loss", "sl_hit", "negative"})
        )
    return pd.Series(pd.NA, index=df.index, dtype="boolean")


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(numeric) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.sum(numeric[mask] * weights[mask]) / np.sum(weights[mask]))


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


@dataclass
class DynamicStrategyState:
    multiplier: float = 1.0
    expected_hit_rate: float = float("nan")
    recent_hit_rate: float = float("nan")
    recent_n: int = 0
    reason: str = "not_refreshed"


class StrategyPerformanceMonitor:
    """Read-only live monitor that returns threshold multipliers."""

    def __init__(
        self,
        *,
        data_root: str,
        run_id: str,
        live_data_root: str,
        ledger_path: str | Path,
        lookback_days: int = 21,
        top_fraction: float = 0.40,
        min_resolved: int = 20,
    ):
        self.data_root = Path(data_root)
        self.run_id = str(run_id)
        self.live_data_root = Path(live_data_root)
        self.ledger_path = Path(ledger_path)
        self.lookback_days = int(lookback_days)
        self.top_fraction = float(top_fraction)
        self.min_resolved = int(min_resolved)
        self.output_path = self.live_data_root / DYNAMIC_PERFORMANCE_PATH
        self._state: dict[tuple[str, str], DynamicStrategyState] = {}
        self._last_refresh: Optional[pd.Timestamp] = None
        self._baseline_candidates: Optional[pd.DataFrame] = None
        self._policy_expected: Optional[dict[str, float]] = None

    def threshold_multiplier(self, strategy_id: str, meta_hash: str | None) -> DynamicStrategyState:
        key = (strategy_core_id(str(strategy_id)), str(meta_hash or "unknown"))
        return self._state.get(key, DynamicStrategyState(reason="neutral_no_dynamic_state"))

    def refresh(self, *, now: Optional[pd.Timestamp] = None, force: bool = False) -> dict[str, Any]:
        now_ts = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")
        if (
            not force
            and self._last_refresh is not None
            and (now_ts - self._last_refresh).total_seconds() < 300.0
        ):
            return self._read_report()
        self._last_refresh = now_ts

        ledger = self._load_ledger()
        baseline = self._load_baseline_candidates()
        policy_expected = self._load_policy_expected_hit_rates()
        report: dict[str, Any] = {
            "updated_at": now_ts.isoformat(),
            "lookback_days": self.lookback_days,
            "top_fraction": self.top_fraction,
            "ledger_path": str(self.ledger_path),
            "baseline_rows": int(len(baseline)),
            "history_backfill_required": True,
            "parity_loading_checker": {
                "sample_rate": 0.05,
                "status": "not_run_in_live_loop",
                "reason": "historical backfill/parity replay must be run by the offline parity job",
            },
            "strategies": {},
        }
        self._state = {}
        if ledger.empty:
            report["reason"] = "empty_prediction_ledger"
            _write_json_atomic(self.output_path, report)
            return report

        ts_col = "signal_bar_ts" if "signal_bar_ts" in ledger.columns else "timestamp"
        ledger[ts_col] = pd.to_datetime(ledger[ts_col], utc=True, errors="coerce")
        recent = ledger.loc[
            ledger[ts_col].notna()
            & (ledger[ts_col] >= now_ts - pd.Timedelta(days=self.lookback_days))
            & (ledger[ts_col] <= now_ts)
        ].copy()
        report["history_backfill_required"] = bool(
            recent.empty
            or recent[ts_col].min() > now_ts - pd.Timedelta(days=self.lookback_days - 1)
        )
        report["recent_rows"] = int(len(recent))
        if recent.empty:
            report["reason"] = "no_recent_prediction_ledger_rows"
            _write_json_atomic(self.output_path, report)
            return report
        if "meta_head_hash" not in recent.columns:
            recent["meta_head_hash"] = "unknown"
        recent["meta_head_hash"] = (
            recent["meta_head_hash"]
            .where(recent["meta_head_hash"].notna(), "unknown")
            .astype(str)
            .replace({"nan": "unknown", "None": "unknown", "": "unknown"})
        )
        if "strategy_id" not in recent.columns:
            report["reason"] = "ledger_missing_strategy_id"
            _write_json_atomic(self.output_path, report)
            return report
        recent["strategy_core"] = recent["strategy_id"].map(lambda x: strategy_core_id(str(x)))

        for (strategy_id, meta_hash), grp in recent.groupby(["strategy_core", "meta_head_hash"], dropna=False):
            grp = grp.copy()
            expected_hit = policy_expected.get(str(strategy_id), float("nan"))
            base_grp = baseline
            if not base_grp.empty and "strategy_core" in base_grp.columns:
                base_grp = base_grp.loc[base_grp["strategy_core"] == str(strategy_id)]
            if not np.isfinite(expected_hit) and not base_grp.empty:
                expected_hit = self._baseline_hit_at_top_fraction(base_grp)
            hit = _outcome_hit(grp)
            resolved = hit.notna()
            weights = _time_decay_weights(grp[ts_col], now_ts) * _rank_weights(grp)
            recent_hit = _weighted_mean(hit.astype(float), weights)
            recent_n = int(resolved.sum())
            if not np.isfinite(expected_hit):
                multiplier = 1.0
                reason = "neutral_missing_oos_baseline_hit_rate"
            elif recent_n < self.min_resolved or not np.isfinite(recent_hit):
                multiplier = 1.0
                reason = "neutral_insufficient_resolved_recent_outcomes"
            else:
                multiplier = float(np.clip(expected_hit / max(recent_hit, 1e-6), 0.8, 1.2))
                reason = "recent_hit_vs_oos_baseline"
            drift = self._drift_metrics(grp, base_grp, now_ts=now_ts)
            for days in (7, 21):
                win = grp.loc[
                    pd.to_datetime(grp[ts_col], utc=True, errors="coerce")
                    >= now_ts - pd.Timedelta(days=days)
                ]
                win_hit = _outcome_hit(win)
                win_weights = _time_decay_weights(win[ts_col], now_ts) * _rank_weights(win)
                achieved = _weighted_mean(win_hit.astype(float), win_weights)
                ratio = (
                    float(achieved / max(expected_hit, 1e-9))
                    if np.isfinite(achieved) and np.isfinite(expected_hit)
                    else float("nan")
                )
                drift[f"dynamic_performance_achieved_hit_rate_{days}d"] = achieved
                drift[f"dynamic_performance_expected_hit_rate_{days}d"] = float(expected_hit)
                drift[f"dynamic_performance_hit_ratio_{days}d"] = ratio
                drift[f"dynamic_performance_surprise_{days}d"] = (
                    float(achieved - expected_hit)
                    if np.isfinite(achieved) and np.isfinite(expected_hit)
                    else float("nan")
                )
                drift[f"dynamic_performance_calibration_{days}d"] = (
                    float(expected_hit / max(achieved, 1e-9))
                    if np.isfinite(achieved) and np.isfinite(expected_hit)
                    else float("nan")
                )
            state = DynamicStrategyState(
                multiplier=multiplier,
                expected_hit_rate=float(expected_hit),
                recent_hit_rate=float(recent_hit),
                recent_n=recent_n,
                reason=reason,
            )
            key = (str(strategy_id), str(meta_hash or "unknown"))
            self._state[key] = state
            report["strategies"][f"{strategy_id}|{meta_hash}"] = {
                "strategy_id": str(strategy_id),
                "meta_head_hash": str(meta_hash or "unknown"),
                "threshold_multiplier": multiplier,
                "expected_hit_rate_oos_top40": float(expected_hit),
                "recent_weighted_hit_rate_21d": float(recent_hit),
                "recent_resolved_n_21d": recent_n,
                "recent_logged_n_21d": int(len(grp)),
                "reason": reason,
                **drift,
            }
        _write_json_atomic(self.output_path, report)
        summary = ", ".join(
            f"{v['strategy_id']}:{v['threshold_multiplier']:.3f}/{v['reason']}"
            for v in list(report["strategies"].values())[:8]
        )
        tprint(
            "Dynamic strategy performance refreshed: "
            f"recent_rows={report.get('recent_rows', 0)} "
            f"baseline_rows={report.get('baseline_rows', 0)} "
            f"history_backfill_required={report.get('history_backfill_required')} "
            f"{summary}"
        )
        return report

    def _read_report(self) -> dict[str, Any]:
        try:
            if self.output_path.exists():
                return json.loads(self.output_path.read_text())
        except Exception:
            pass
        return {}

    def _load_ledger(self) -> pd.DataFrame:
        try:
            if self.ledger_path.exists():
                return pd.read_parquet(self.ledger_path)
        except Exception as exc:
            tprint(f"Dynamic strategy performance: could not read ledger {self.ledger_path}: {exc}")
        return pd.DataFrame()

    def _candidate_paths(self) -> list[Path]:
        base = self.data_root / "artifacts" / self.run_id
        return [
            base / "policy_params" / "simple_policy_candidates.parquet",
            base / "simple_policy_optimiser" / "simple_policy_candidates.parquet",
            base / "simple_policy_optimiser" / "deployment" / "simple_policy_candidates.parquet",
            base / "portfolio_policy_replay" / "per_candidate_replay_decisions.parquet",
            base / "simple_policy_candidates.parquet",
            base / "policy_params" / "auction_rank_reference.parquet",
        ]

    def _load_baseline_candidates(self) -> pd.DataFrame:
        if self._baseline_candidates is not None:
            return self._baseline_candidates
        path = _first_existing(self._candidate_paths())
        if path is None:
            self._baseline_candidates = pd.DataFrame()
            return self._baseline_candidates
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            tprint(f"Dynamic strategy performance: could not read baseline candidates {path}: {exc}")
            self._baseline_candidates = pd.DataFrame()
            return self._baseline_candidates
        if "strategy_id" in df.columns:
            df["strategy_core"] = df["strategy_id"].map(lambda x: strategy_core_id(str(x)))
        self._baseline_candidates = df
        return df

    def _policy_paths(self) -> list[Path]:
        base = self.data_root / "artifacts" / self.run_id
        return [
            base / "policy_params" / "best_policy_params.json",
            base / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
            base / "simple_policy_optimiser" / "best_policy_params.json",
            base / "best_policy_params.json",
        ]

    def _load_policy_expected_hit_rates(self) -> dict[str, float]:
        if self._policy_expected is not None:
            return self._policy_expected
        out: dict[str, float] = {}
        path = _first_existing(self._policy_paths())
        if path is None:
            self._policy_expected = out
            return out
        try:
            data = json.loads(path.read_text())
        except Exception:
            self._policy_expected = out
            return out
        rows = []
        if isinstance(data, dict):
            if isinstance(data.get("strategies"), list):
                rows = data.get("strategies") or []
            elif isinstance(data.get("strategy_params"), dict):
                rows = list((data.get("strategy_params") or {}).values())
            else:
                rows = [
                    value
                    for value in data.values()
                    if isinstance(value, dict)
                ]
        for row in rows:
            sid = row.get("strategy_id") or row.get("strategy") or row.get("strategy_core")
            hit = row.get("pnl_positive_rate", row.get("hit_rate", row.get("win_rate")))
            if sid is None:
                continue
            hit_f = _finite_float(hit)
            if np.isfinite(hit_f):
                out[strategy_core_id(str(sid))] = hit_f
        self._policy_expected = out
        return out

    def _baseline_hit_at_top_fraction(self, df: pd.DataFrame) -> float:
        rank_col = next(
            (c for c in ("strategy_rank_pct", "policy_rank_pct", "normalized_rank_score") if c in df.columns),
            None,
        )
        ret_col = next((c for c in ("net_return", "net_pnl_pct", "return") if c in df.columns), None)
        if rank_col is None or ret_col is None:
            return float("nan")
        rank = pd.to_numeric(df[rank_col], errors="coerce")
        ret = pd.to_numeric(df[ret_col], errors="coerce")
        keep = rank >= float(1.0 - self.top_fraction)
        vals = ret.loc[keep & ret.notna()]
        if vals.empty:
            return float("nan")
        return float((vals > 0.0).mean())

    def _drift_metrics(self, recent: pd.DataFrame, baseline: pd.DataFrame, *, now_ts: pd.Timestamp) -> dict[str, Any]:
        def _mean_col(frame: pd.DataFrame, *cols: str) -> float:
            for col in cols:
                if col not in frame.columns:
                    continue
                vals = pd.to_numeric(frame[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                vals = vals.dropna()
                if not vals.empty:
                    return float(vals.mean())
            return float("nan")

        def _delta_col(*cols: str) -> float:
            recent_val = _mean_col(recent, *cols)
            base_val = _mean_col(baseline, *cols)
            if np.isfinite(recent_val) and np.isfinite(base_val):
                return float(recent_val - base_val)
            return float("nan")

        score_col_recent = next((c for c in ("calibrated_score", "meta_pred", "raw_prediction_score") if c in recent.columns), None)
        score_col_base = next((c for c in ("calibrated_score", "meta_pred", "score") if c in baseline.columns), None)
        rank_col_recent = next((c for c in ("policy_rank_pct", "normalized_rank_score", "meta_train_rank_pct") if c in recent.columns), None)
        rank_col_base = next((c for c in ("strategy_rank_pct", "policy_rank_pct", "normalized_rank_score") if c in baseline.columns), None)
        prediction_score_psi = (
            _psi(recent[score_col_recent], baseline[score_col_base])
            if score_col_recent and score_col_base
            else float("nan")
        )
        raw_logit_psi = float("nan")
        if score_col_recent and score_col_base:
            r = pd.to_numeric(recent[score_col_recent], errors="coerce").clip(1e-6, 1 - 1e-6)
            b = pd.to_numeric(baseline[score_col_base], errors="coerce").clip(1e-6, 1 - 1e-6)
            raw_logit_psi = _psi(np.log(r / (1.0 - r)), np.log(b / (1.0 - b)))
        rank_pct_psi = (
            _psi(recent[rank_col_recent], baseline[rank_col_base])
            if rank_col_recent and rank_col_base
            else float("nan")
        )
        topq_threshold_drift = float("nan")
        topq_candidate_count_drift = float("nan")
        if rank_col_recent and rank_col_base:
            rr = pd.to_numeric(recent[rank_col_recent], errors="coerce").dropna()
            br = pd.to_numeric(baseline[rank_col_base], errors="coerce").dropna()
            if len(rr) and len(br):
                topq_threshold_drift = float(rr.quantile(1.0 - self.top_fraction) - br.quantile(1.0 - self.top_fraction))
                recent_days = max((now_ts - pd.to_datetime(recent.get("signal_bar_ts", recent.get("timestamp")), utc=True, errors="coerce").min()).total_seconds() / 86400.0, 1.0)
                recent_rate = float((rr >= rr.quantile(1.0 - self.top_fraction)).sum() / recent_days)
                base_rate = float((br >= br.quantile(1.0 - self.top_fraction)).sum() / max(len(br) / 24.0, 1.0))
                topq_candidate_count_drift = float(recent_rate / max(base_rate, 1e-9) - 1.0)
        strategy_mix_drift = _jsd(recent["strategy_core"], baseline["strategy_core"]) if "strategy_core" in baseline.columns else float("nan")
        symbol_mix_drift = _jsd(recent["symbol"], baseline["symbol"]) if "symbol" in recent.columns and "symbol" in baseline.columns else float("nan")
        score_uncertainty = float("nan")
        if score_col_recent:
            p = pd.to_numeric(recent[score_col_recent], errors="coerce")
            score_uncertainty = float((1.0 - (p - 0.5).abs() * 2.0).replace([np.inf, -np.inf], np.nan).mean())
        logged_uncertainty = _mean_col(recent, "uncertainty_score", "prob_uncertainty")
        uncertainty_score = (
            logged_uncertainty if np.isfinite(logged_uncertainty) else score_uncertainty
        )
        feature_drift_psi = _mean_col(
            recent,
            "feature_drift_psi_core_50",
            "feature_drift_psi_core",
        )
        feature_drift_ks = _mean_col(
            recent,
            "feature_drift_ks_core",
            "feature_drift_ks_bin_mean",
        )
        feature_drift_cov_shift = _mean_col(recent, "feature_drift_cov_shift")
        contribution_drift_jsd = _delta_col("contrib_balance")
        contrib_top1_abs_share_drift = _delta_col("contrib_top1_abs_share")
        contrib_entropy_drift = _delta_col("contrib_entropy")
        rare_leaf_fraction_drift = _delta_col("rare_leaf_fraction")
        leaf_support_drift = _delta_col("leaf_count_p10")
        regime_centroid_similarity = _mean_col(
            recent, "regime_centroid_similarity_train"
        )
        regime_centroid_similarity_drift = (
            float(1.0 - regime_centroid_similarity)
            if np.isfinite(regime_centroid_similarity)
            else float("nan")
        )
        drift_parts = [
            prediction_score_psi,
            rank_pct_psi,
            abs(topq_threshold_drift),
            feature_drift_psi,
            feature_drift_ks,
            feature_drift_cov_shift,
            abs(rare_leaf_fraction_drift),
            abs(leaf_support_drift),
        ]
        drift_parts = [float(x) for x in drift_parts if np.isfinite(float(x))]
        inference_drift_score = (
            float(np.mean(drift_parts)) if drift_parts else float("nan")
        )
        return {
            "feature_drift_psi_core_50": feature_drift_psi,
            "feature_drift_psi_core_80": _mean_col(
                recent, "feature_drift_psi_core_80", "feature_drift_psi_core"
            ),
            "feature_drift_ks_core": feature_drift_ks,
            "feature_drift_cov_shift": feature_drift_cov_shift,
            "prediction_score_psi": prediction_score_psi,
            "raw_logit_psi": raw_logit_psi,
            "rank_pct_psi": rank_pct_psi,
            "topq_threshold_drift": topq_threshold_drift,
            "topq_candidate_count_drift": topq_candidate_count_drift,
            "topq_strategy_mix_drift": strategy_mix_drift,
            "topq_symbol_mix_drift": symbol_mix_drift,
            "topq_regime_mix_drift": float("nan"),
            "contribution_drift_jsd": contribution_drift_jsd,
            "contrib_top1_abs_share_drift": contrib_top1_abs_share_drift,
            "contrib_entropy_drift": contrib_entropy_drift,
            "rare_leaf_fraction_drift": rare_leaf_fraction_drift,
            "leaf_support_drift": leaf_support_drift,
            "regime_centroid_similarity_drift": regime_centroid_similarity_drift,
            "z_prob_uncertainty": score_uncertainty,
            "z_leaf_or_support_uncertainty": _mean_col(recent, "rare_leaf_fraction"),
            "z_contribution_uncertainty": _mean_col(recent, "contrib_entropy"),
            "z_regime_distance": regime_centroid_similarity_drift,
            "uncertainty_score": uncertainty_score,
            "uncertainty_score_ratio_7d": float("nan"),
            "uncertainty_score_ratio_21d": float("nan"),
            "inference_drift_score": inference_drift_score,
            "inference_drift_score_7d": inference_drift_score,
            "inference_drift_score_21d": inference_drift_score,
            "drift_component_status": {
                "feature_drift": (
                    "available_from_prediction_ledger"
                    if np.isfinite(feature_drift_psi)
                    or np.isfinite(feature_drift_cov_shift)
                    else "not_available_from_prediction_ledger"
                ),
                "contribution_drift": (
                    "available_from_prediction_ledger"
                    if any(
                        np.isfinite(x)
                        for x in (
                            contribution_drift_jsd,
                            contrib_top1_abs_share_drift,
                            contrib_entropy_drift,
                        )
                    )
                    else "not_available_from_prediction_ledger"
                ),
                "leaf_support": (
                    "available_from_prediction_ledger"
                    if np.isfinite(rare_leaf_fraction_drift)
                    or np.isfinite(leaf_support_drift)
                    else "not_available_from_prediction_ledger"
                ),
                "uncertainty": (
                    "lgbm_uncertainty_logged"
                    if np.isfinite(logged_uncertainty)
                    else "score_distance_proxy_only_until_lgbm_uncertainty_logged"
                ),
            },
        }
