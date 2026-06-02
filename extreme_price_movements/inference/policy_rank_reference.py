from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.inference.parity import strategy_core_id

POLICY_RANK_REFERENCE_SCHEMA_VERSION = "policy_rank_reference_v1"
POLICY_RANK_REFERENCE_GENERATOR = "simple_policy_optimiser"
POLICY_RANK_REFERENCE_DIR = "rank_reference"
POLICY_RANK_REFERENCE_SCORE_COL = "calibrated_score"
POLICY_RANK_REFERENCE_RANK_COL = "rank_pct"
AUCTION_RANK_REFERENCE_FILE = "cross_strategy_auction.parquet"


def _constant_string_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    values = (
        frame[column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    values = values[values != ""].drop_duplicates()
    if len(values) == 1:
        return str(values.iloc[0])
    return ""


def _policy_oos_contract_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    generation_source = _constant_string_value(frame, "policy_oos_generation_source")
    source_fit_end = _constant_string_value(frame, "policy_oos_source_model_fit_end")
    if not generation_source and not source_fit_end:
        return {}
    return {
        "schema_version": "policy_rank_reference_policy_oos_contract_v1",
        "policy_oos_generation_source": generation_source or None,
        "policy_oos_source_model_fit_end": source_fit_end or None,
        "rank_normalization": "policy_rank_reference_percentile_from_policy_oos_clf",
    }


def _policy_oos_contract_valid(entry: dict[str, Any]) -> bool:
    contract = entry.get("policy_oos_contract")
    if not isinstance(contract, dict) or not contract:
        # Legacy rank references did not persist scorer provenance. They remain
        # readable until regenerated, but regenerated references fail closed if
        # their explicit contract is wrong.
        return True
    source = str(contract.get("policy_oos_generation_source") or "")
    rank_norm = str(contract.get("rank_normalization") or "")
    return bool(
        source.startswith("generated_from_train_meta_state")
        and "policy_rank_reference_percentile" in rank_norm
    )


def _manifest_policy_oos_contract(
    strategies: dict[str, Any],
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    for entry in strategies.values():
        if not isinstance(entry, dict):
            continue
        contract = entry.get("policy_oos_contract")
        if isinstance(contract, dict) and contract:
            contracts.append(dict(contract))
    if not contracts:
        return {}
    first = contracts[0]
    if all(contract == first for contract in contracts[1:]):
        return first
    return {
        "schema_version": "policy_rank_reference_policy_oos_contract_v1",
        "policy_oos_generation_source": "generated_from_train_meta_state:mixed",
        "policy_oos_source_model_fit_end": "mixed",
        "rank_normalization": "policy_rank_reference_percentile_from_policy_oos_clf",
        "strategy_contract_count": len(contracts),
    }


def _safe_strategy_filename(strategy_id: str) -> str:
    """Keep artifact names readable while preventing path separators."""
    sid = str(strategy_id or "").strip()
    sid = re.sub(r"[\\/]+", "_", sid)
    sid = re.sub(r"[^A-Za-z0-9_.=-]+", "_", sid)
    return sid or "unknown_strategy"


def _rank_reference_root(data_root: str | Path, run_id: str) -> Path:
    return (
        Path(data_root)
        / "artifacts"
        / str(run_id)
        / "simple_policy_optimiser"
        / POLICY_RANK_REFERENCE_DIR
    )


def policy_rank_pct_from_sorted_scores(
    sorted_scores: Iterable[float] | np.ndarray,
    calibrated_score: float,
) -> float:
    """Map a score into the saved policy-rank percentile CDF."""
    if isinstance(sorted_scores, np.ndarray):
        scores = np.asarray(sorted_scores, dtype=np.float64)
    else:
        scores = np.asarray(list(sorted_scores), dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0 or not np.isfinite(float(calibrated_score)):
        return float("nan")
    scores.sort()
    return float(
        np.searchsorted(scores, float(calibrated_score), side="right") / scores.size
    )


def strategy_rank_reference_aliases(
    strategy_id: str, side: str | None = None
) -> list[str]:
    sid = str(strategy_id or "").strip()
    aliases: list[str] = []
    if sid:
        aliases.append(sid)
    core = strategy_core_id(sid)
    if core and core not in aliases:
        aliases.append(core)
    side_s = str(side or "").strip().lower()
    sides = [side_s] if side_s in {"long", "short"} else ["long", "short"]
    for prefix in sides:
        candidate = f"{prefix}_{core}" if core else ""
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def persist_policy_rank_reference(
    df_policy_all: pd.DataFrame,
    *,
    data_root: str | Path,
    run_id: str,
    strategy_id: str,
    market_mode: str | None = None,
) -> Path:
    """Persist the exact policy-slice rank population used by Stage A."""
    required = {POLICY_RANK_REFERENCE_SCORE_COL, POLICY_RANK_REFERENCE_RANK_COL}
    missing = sorted(required.difference(df_policy_all.columns))
    if missing:
        raise ValueError(f"policy rank reference missing required columns: {missing}")

    out_dir = _rank_reference_root(data_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = str(strategy_id)
    file_name = f"{_safe_strategy_filename(sid)}.parquet"
    out_path = out_dir / file_name

    cols = [
        "strategy_id",
        POLICY_RANK_REFERENCE_SCORE_COL,
        POLICY_RANK_REFERENCE_RANK_COL,
    ]
    for optional_col in ("timestamp", "symbol", "market_mode"):
        if optional_col in df_policy_all.columns and optional_col not in cols:
            cols.append(optional_col)
    ref = df_policy_all.copy()
    ref["strategy_id"] = sid
    if market_mode is not None and "market_mode" not in ref.columns:
        ref["market_mode"] = str(market_mode)
        if "market_mode" not in cols:
            cols.append("market_mode")
    ref = ref[cols].copy()
    ref[POLICY_RANK_REFERENCE_SCORE_COL] = pd.to_numeric(
        ref[POLICY_RANK_REFERENCE_SCORE_COL], errors="coerce"
    )
    ref[POLICY_RANK_REFERENCE_RANK_COL] = pd.to_numeric(
        ref[POLICY_RANK_REFERENCE_RANK_COL], errors="coerce"
    )
    ref = ref.dropna(
        subset=[POLICY_RANK_REFERENCE_SCORE_COL, POLICY_RANK_REFERENCE_RANK_COL]
    )
    if ref.empty:
        raise ValueError(f"policy rank reference for {sid} has no finite rows")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    ref.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "schema_version": POLICY_RANK_REFERENCE_SCHEMA_VERSION,
            "generated_by": POLICY_RANK_REFERENCE_GENERATOR,
            "run_id": str(run_id),
            "market_mode": str(market_mode or ""),
        }
    )
    strategies = dict(manifest.get("strategies") or {})
    scores = ref[POLICY_RANK_REFERENCE_SCORE_COL].to_numpy(dtype=np.float64)
    strategies[sid] = {
        "path": str(out_path.relative_to(Path(data_root) / "artifacts" / str(run_id))),
        "n_rows": int(len(ref)),
        "score_col": POLICY_RANK_REFERENCE_SCORE_COL,
        "rank_col": POLICY_RANK_REFERENCE_RANK_COL,
        "min_score": float(np.nanmin(scores)),
        "max_score": float(np.nanmax(scores)),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    policy_oos_contract = _policy_oos_contract_from_frame(df_policy_all)
    if policy_oos_contract:
        strategies[sid]["policy_oos_contract"] = policy_oos_contract
    manifest["strategies"] = strategies
    manifest_policy_contract = _manifest_policy_oos_contract(strategies)
    if manifest_policy_contract:
        manifest["policy_oos_contract"] = manifest_policy_contract
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_manifest, manifest_path)
    return out_path


def persist_auction_rank_reference(
    candidates: pd.DataFrame,
    *,
    data_root: str | Path,
    run_id: str,
    market_mode: str | None = None,
    score_col: str = POLICY_RANK_REFERENCE_SCORE_COL,
) -> Path:
    """Persist the cross-strategy score population used by portfolio auction."""
    if score_col not in candidates.columns:
        raise ValueError(f"auction rank reference missing score column: {score_col}")
    out_dir = _rank_reference_root(data_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / AUCTION_RANK_REFERENCE_FILE
    ref = candidates.copy()
    ref[score_col] = pd.to_numeric(ref[score_col], errors="coerce")
    ref = ref.dropna(subset=[score_col])
    if ref.empty:
        raise ValueError("auction rank reference has no finite rows")
    cols = [score_col]
    for optional_col in (
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "strategy_rank_pct",
        "normalized_rank_score",
        "market_mode",
    ):
        if optional_col in ref.columns and optional_col not in cols:
            cols.append(optional_col)
    if market_mode is not None and "market_mode" not in ref.columns:
        ref["market_mode"] = str(market_mode)
        cols.append("market_mode")
    ref = ref[cols].copy()
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    ref.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "schema_version": POLICY_RANK_REFERENCE_SCHEMA_VERSION,
            "generated_by": POLICY_RANK_REFERENCE_GENERATOR,
            "run_id": str(run_id),
            "market_mode": str(market_mode or ""),
        }
    )
    scores = ref[score_col].to_numpy(dtype=np.float64)
    manifest["auction"] = {
        "path": str(out_path.relative_to(Path(data_root) / "artifacts" / str(run_id))),
        "n_rows": int(len(ref)),
        "score_col": score_col,
        "rank_col": "normalized_rank_score",
        "min_score": float(np.nanmin(scores)),
        "max_score": float(np.nanmax(scores)),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_manifest, manifest_path)
    return out_path


def invalidate_auction_rank_reference(
    *,
    data_root: str | Path,
    run_id: str,
    market_mode: str | None = None,
    reason: str,
) -> None:
    """Clear stale cross-strategy auction manifest entries before a fresh export."""
    out_dir = _rank_reference_root(data_root, run_id)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(manifest, dict):
        return
    previous = manifest.pop("auction", None)
    previous_strategies = manifest.pop("strategies", None)
    manifest.update(
        {
            "schema_version": POLICY_RANK_REFERENCE_SCHEMA_VERSION,
            "generated_by": POLICY_RANK_REFERENCE_GENERATOR,
            "run_id": str(run_id),
            "market_mode": str(market_mode or manifest.get("market_mode") or ""),
            "auction_invalidated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "auction_invalidated_reason": str(reason),
        }
    )
    if previous is not None:
        manifest["previous_auction"] = previous
    if isinstance(previous_strategies, dict):
        manifest["previous_strategy_count"] = len(previous_strategies)
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_manifest, manifest_path)


@dataclass(frozen=True)
class PolicyRankLookupResult:
    policy_rank_pct: float
    n_rows: int
    source: str
    strategy_id: str


@dataclass(frozen=True)
class AuctionEvThresholdResult:
    threshold: float
    target_mean_net_return: float
    target_hit_rate: float
    mean_net_return: float
    hit_rate: float
    n_trades: int
    source: str
    enabled: bool = True
    reason: str = ""


@dataclass(frozen=True)
class StrategyEvGateResult:
    allowed: bool
    target_mean_net_return: float
    min_hit_rate: float
    mean_net_return: float
    hit_rate: float
    source: str
    reason: str = ""


class PolicyRankReferenceStore:
    """Lazy loader for simple_policy_optimiser policy-rank CDF artifacts."""

    def __init__(self, *, data_root: str | Path, run_id: str):
        self.data_root = Path(data_root)
        self.run_id = str(run_id)
        self.root = _rank_reference_root(self.data_root, self.run_id)
        self.manifest_path = self.root / "manifest.json"
        self._manifest: dict[str, Any] | None = None
        self._cache: dict[str, tuple[np.ndarray, str, str]] = {}
        self._auction_cache: tuple[np.ndarray, str] | None = None
        self._auction_ev_threshold_table: pd.DataFrame | None = None
        self._strategy_ev_threshold_tables: dict[str, pd.DataFrame] | None = None
        self._strategy_ev_gate_table: dict[str, dict[str, Any]] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            try:
                self._manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._manifest = {}
        return self._manifest

    def _strategy_entry(
        self, strategy_id: str, side: str | None = None
    ) -> tuple[str, dict[str, Any]] | tuple[None, None]:
        strategies = self.manifest.get("strategies") or {}
        for alias in strategy_rank_reference_aliases(strategy_id, side):
            entry = strategies.get(alias)
            if isinstance(entry, dict):
                return alias, entry
        return None, None

    def _load_scores(
        self, strategy_id: str, side: str | None = None
    ) -> tuple[np.ndarray, str, str] | None:
        alias, entry = self._strategy_entry(strategy_id, side)
        if not alias or not isinstance(entry, dict):
            return None
        manifest_contract = self.manifest.get("policy_oos_contract")
        if isinstance(manifest_contract, dict) and manifest_contract:
            if not _policy_oos_contract_valid(
                {"policy_oos_contract": manifest_contract}
            ):
                return None
        if not _policy_oos_contract_valid(entry):
            return None
        if alias in self._cache:
            return self._cache[alias]
        rel_path = str(entry.get("path") or "")
        path = self.data_root / "artifacts" / self.run_id / rel_path
        score_col = str(entry.get("score_col") or POLICY_RANK_REFERENCE_SCORE_COL)
        try:
            frame = pd.read_parquet(path, columns=[score_col])
            scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(
                dtype=np.float64
            )
        except Exception:
            return None
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return None
        scores.sort()
        source = str(path)
        loaded = (scores, source, alias)
        self._cache[alias] = loaded
        return loaded

    def lookup(
        self,
        *,
        strategy_id: str,
        calibrated_score: float,
        side: str | None = None,
    ) -> PolicyRankLookupResult:
        loaded = self._load_scores(strategy_id, side)
        if loaded is None:
            return PolicyRankLookupResult(float("nan"), 0, "", "")
        scores, source, alias = loaded
        rank = policy_rank_pct_from_sorted_scores(scores, float(calibrated_score))
        return PolicyRankLookupResult(rank, int(scores.size), source, alias)

    def _load_auction_scores(self) -> tuple[np.ndarray, str] | None:
        if self._auction_cache is not None:
            return self._auction_cache
        entry = self.manifest.get("auction")
        if not isinstance(entry, dict):
            return None
        rel_path = str(entry.get("path") or "")
        path = self.data_root / "artifacts" / self.run_id / rel_path
        score_col = str(entry.get("score_col") or POLICY_RANK_REFERENCE_SCORE_COL)
        try:
            frame = pd.read_parquet(path, columns=[score_col])
            scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(
                dtype=np.float64
            )
        except Exception:
            return None
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return None
        scores.sort()
        self._auction_cache = (scores, str(path))
        return self._auction_cache

    def lookup_auction(
        self,
        *,
        calibrated_score: float,
    ) -> PolicyRankLookupResult:
        loaded = self._load_auction_scores()
        if loaded is None:
            return PolicyRankLookupResult(float("nan"), 0, "", "cross_strategy")
        scores, source = loaded
        rank = policy_rank_pct_from_sorted_scores(scores, float(calibrated_score))
        return PolicyRankLookupResult(rank, int(scores.size), source, "cross_strategy")

    def _load_auction_ev_threshold_table(self) -> pd.DataFrame:
        """Load or build the global-auction threshold -> EV/hit-rate map."""
        if self._auction_ev_threshold_table is not None:
            return self._auction_ev_threshold_table
        policy_path = (
            self.data_root
            / "artifacts"
            / self.run_id
            / "policy_params"
            / "best_policy_params.json"
        )
        try:
            params = json.loads(policy_path.read_text(encoding="utf-8"))
            rows: list[dict[str, Any]] = []
            for strategy in params.get("strategies") or []:
                metrics = dict(strategy.get("deployment_threshold_metrics") or {})
                search = dict(metrics.get("threshold_search") or {})
                for item in search.get("best_by_threshold") or []:
                    threshold = item.get("deployment_rank_threshold")
                    mean_net = item.get(
                        "cumulative_mean_net_trade", item.get("mean_net_trade")
                    )
                    hit_rate = item.get("cumulative_hit_rate", item.get("hit_rate"))
                    n_trades = item.get("cumulative_n_trades", item.get("n_trades"))
                    try:
                        threshold_f = float(threshold)
                        mean_f = float(mean_net)
                        hit_f = float(hit_rate)
                        n_i = int(n_trades)
                    except (TypeError, ValueError):
                        continue
                    if (
                        np.isfinite(threshold_f)
                        and np.isfinite(mean_f)
                        and np.isfinite(hit_f)
                        and n_i > 0
                    ):
                        rows.append(
                            {
                                "threshold": threshold_f,
                                "mean_net_return": mean_f,
                                "hit_rate": hit_f,
                                "n_trades": n_i,
                                "source": str(policy_path),
                            }
                        )
            if rows:
                raw = pd.DataFrame(rows)
                grouped_rows: list[dict[str, Any]] = []
                for threshold, group in raw.groupby("threshold"):
                    weights = pd.to_numeric(group["n_trades"], errors="coerce").fillna(
                        0.0
                    )
                    weight_sum = float(weights.sum())
                    if weight_sum <= 0:
                        continue
                    grouped_rows.append(
                        {
                            "threshold": float(threshold),
                            "mean_net_return": float(
                                (
                                    pd.to_numeric(
                                        group["mean_net_return"], errors="coerce"
                                    ).fillna(0.0)
                                    * weights
                                ).sum()
                                / weight_sum
                            ),
                            "hit_rate": float(
                                (
                                    pd.to_numeric(group["hit_rate"], errors="coerce")
                                    .fillna(0.0)
                                    .mul(weights)
                                ).sum()
                                / weight_sum
                            ),
                            "n_trades": int(weight_sum),
                            "source": str(policy_path),
                        }
                    )
                if grouped_rows:
                    self._auction_ev_threshold_table = pd.DataFrame(
                        grouped_rows
                    ).sort_values("threshold")
                    return self._auction_ev_threshold_table
        except Exception:
            pass
        path = (
            self.data_root
            / "artifacts"
            / self.run_id
            / "simple_policy_optimiser"
            / "simple_policy_candidates.parquet"
        )
        try:
            frame = pd.read_parquet(
                path,
                columns=["auction_rank_score", "normalized_rank_score", "net_return"],
            )
        except Exception:
            self._auction_ev_threshold_table = pd.DataFrame()
            return self._auction_ev_threshold_table
        rank_col = (
            "auction_rank_score"
            if "auction_rank_score" in frame.columns
            else "normalized_rank_score"
        )
        work = frame[[rank_col, "net_return"]].copy()
        work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce")
        work["net_return"] = pd.to_numeric(work["net_return"], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if work.empty:
            self._auction_ev_threshold_table = pd.DataFrame()
            return self._auction_ev_threshold_table
        thresholds = np.unique(np.round(work[rank_col].to_numpy(dtype=float), 4))
        thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
        rows: list[dict[str, Any]] = []
        returns = work["net_return"].to_numpy(dtype=float)
        ranks = work[rank_col].to_numpy(dtype=float)
        for threshold in thresholds:
            mask = ranks >= float(threshold)
            n = int(mask.sum())
            if n <= 0:
                continue
            selected = returns[mask]
            rows.append(
                {
                    "threshold": float(threshold),
                    "mean_net_return": float(np.nanmean(selected)),
                    "hit_rate": float(np.nanmean(selected > 0.0)),
                    "n_trades": n,
                    "source": str(path),
                }
            )
        self._auction_ev_threshold_table = pd.DataFrame(rows).sort_values(
            "threshold"
        )
        return self._auction_ev_threshold_table

    def auction_threshold_for_ev(
        self,
        *,
        target_mean_net_return: float,
        min_hit_rate: float = 0.60,
        fallback_threshold: float = 1.0,
    ) -> AuctionEvThresholdResult:
        """Return the lowest global-auction floor meeting EV and hit-rate constraints."""
        target = float(target_mean_net_return)
        hit_target = float(min_hit_rate)
        table = self._load_auction_ev_threshold_table()
        if table.empty:
            return AuctionEvThresholdResult(
                threshold=float(fallback_threshold),
                target_mean_net_return=target,
                target_hit_rate=hit_target,
                mean_net_return=float("nan"),
                hit_rate=float("nan"),
                n_trades=0,
                source="",
                enabled=False,
                reason="missing_auction_ev_threshold_table",
            )
        ok = table[
            (pd.to_numeric(table["mean_net_return"], errors="coerce") >= target)
            & (pd.to_numeric(table["hit_rate"], errors="coerce") >= hit_target)
        ]
        if ok.empty:
            best = table.sort_values(
                ["mean_net_return", "hit_rate", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
            return AuctionEvThresholdResult(
                threshold=float(fallback_threshold),
                target_mean_net_return=target,
                target_hit_rate=hit_target,
                mean_net_return=float(best.get("mean_net_return", np.nan)),
                hit_rate=float(best.get("hit_rate", np.nan)),
                n_trades=int(best.get("n_trades", 0)),
                source=str(best.get("source", "")),
                enabled=False,
                reason="no_threshold_meets_ev_and_hit_rate_constraints",
            )
        row = ok.sort_values("threshold").iloc[0]
        return AuctionEvThresholdResult(
            threshold=float(row["threshold"]),
            target_mean_net_return=target,
            target_hit_rate=hit_target,
            mean_net_return=float(row["mean_net_return"]),
            hit_rate=float(row["hit_rate"]),
            n_trades=int(row["n_trades"]),
            source=str(row.get("source", "")),
            enabled=True,
            reason="auction_ev_threshold",
        )

    def _load_strategy_ev_threshold_tables(self) -> dict[str, pd.DataFrame]:
        """Load per-strategy threshold -> EV/hit-rate maps from policy replay output."""
        if self._strategy_ev_threshold_tables is not None:
            return self._strategy_ev_threshold_tables
        policy_path = (
            self.data_root
            / "artifacts"
            / self.run_id
            / "policy_params"
            / "best_policy_params.json"
        )
        deployed_rows: dict[str, dict[str, Any]] = {}
        try:
            params = json.loads(policy_path.read_text(encoding="utf-8"))
            for strategy in params.get("strategies") or []:
                sid = str(strategy.get("strategy_id") or "").strip()
                if not sid:
                    continue
                try:
                    threshold_f = float(strategy.get("deployment_rank_threshold"))
                    mean_f = float(strategy.get("avg_net_pnl_per_trade"))
                    hit_f = float(
                        strategy.get(
                            "pnl_positive_rate",
                            strategy.get("hit_rate"),
                        )
                    )
                except (TypeError, ValueError):
                    continue
                if not (
                    np.isfinite(threshold_f)
                    and np.isfinite(mean_f)
                    and np.isfinite(hit_f)
                ):
                    continue
                try:
                    n_i = int(
                        strategy.get(
                            "trade_count",
                            strategy.get(
                                "n_trades",
                                strategy.get(
                                    "deployment_trades",
                                    strategy.get("candidate_rows", 0),
                                ),
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    n_i = 0
                deployed_rows[sid] = {
                    "threshold": float(np.clip(threshold_f, 0.0, 1.0)),
                    "mean_net_return": mean_f,
                    "hit_rate": hit_f,
                    "n_trades": max(0, n_i),
                    "source": str(policy_path),
                }
        except Exception:
            deployed_rows = {}
        candidates_path = (
            self.data_root
            / "artifacts"
            / self.run_id
            / "simple_policy_optimiser"
            / "simple_policy_candidates.parquet"
        )
        try:
            frame = pd.read_parquet(
                candidates_path,
                columns=["strategy_id", "side", "strategy_rank_pct", "net_return"],
            )
            frame["strategy_id"] = frame["strategy_id"].astype(str)
            frame["side"] = frame["side"].astype(str).str.lower()
            frame["strategy_rank_pct"] = pd.to_numeric(
                frame["strategy_rank_pct"], errors="coerce"
            )
            frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["strategy_id", "strategy_rank_pct", "net_return"]
            )
            frame = frame[
                (frame["strategy_rank_pct"] >= 0.0)
                & (frame["strategy_rank_pct"] <= 1.0)
            ]
            out: dict[str, pd.DataFrame] = {}
            for sid, group in frame.groupby("strategy_id", sort=False):
                ranks = group["strategy_rank_pct"].to_numpy(dtype=np.float64)
                returns = group["net_return"].to_numpy(dtype=np.float64)
                if ranks.size == 0:
                    continue
                thresholds = np.unique(np.round(ranks, 4))
                thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
                rows: list[dict[str, Any]] = []
                for threshold in thresholds:
                    mask = ranks >= float(threshold)
                    n = int(mask.sum())
                    if n <= 0:
                        continue
                    selected = returns[mask]
                    rows.append(
                        {
                            "threshold": float(threshold),
                            "mean_net_return": float(np.nanmean(selected)),
                            "hit_rate": float(np.nanmean(selected > 0.0)),
                            "n_trades": n,
                            "source": str(candidates_path),
                        }
                    )
                deployed = deployed_rows.get(str(sid))
                if deployed is not None:
                    rows.append(deployed)
                if not rows:
                    continue
                side = None
                side_values = [
                    str(x).strip().lower()
                    for x in group["side"].dropna().unique().tolist()
                ]
                if len(side_values) == 1 and side_values[0] in {"long", "short"}:
                    side = side_values[0]
                table = (
                    pd.DataFrame(rows)
                    .sort_values(
                        ["threshold", "mean_net_return", "hit_rate"],
                        ascending=[True, False, False],
                    )
                    .drop_duplicates(subset=["threshold"], keep="first")
                    .sort_values("threshold")
                )
                for alias in strategy_rank_reference_aliases(str(sid), side):
                    if alias:
                        out[str(alias)] = table
            if out:
                self._strategy_ev_threshold_tables = out
                return self._strategy_ev_threshold_tables
        except Exception:
            pass
        out: dict[str, pd.DataFrame] = {}
        try:
            params = json.loads(policy_path.read_text(encoding="utf-8"))
            for strategy in params.get("strategies") or []:
                sid = str(strategy.get("strategy_id") or "").strip()
                if not sid:
                    continue
                side = str(strategy.get("side") or "").strip().lower() or None
                metrics = dict(strategy.get("deployment_threshold_metrics") or {})
                search = dict(metrics.get("threshold_search") or {})
                rows: list[dict[str, Any]] = []
                for item in search.get("best_by_threshold") or []:
                    threshold = item.get("deployment_rank_threshold")
                    mean_net = item.get(
                        "cumulative_mean_net_trade", item.get("mean_net_trade")
                    )
                    hit_rate = item.get("cumulative_hit_rate", item.get("hit_rate"))
                    n_trades = item.get("cumulative_n_trades", item.get("n_trades"))
                    try:
                        threshold_f = float(threshold)
                        mean_f = float(mean_net)
                        hit_f = float(hit_rate)
                        n_i = int(n_trades)
                    except (TypeError, ValueError):
                        continue
                    if (
                        np.isfinite(threshold_f)
                        and np.isfinite(mean_f)
                        and np.isfinite(hit_f)
                        and n_i > 0
                    ):
                        rows.append(
                            {
                                "threshold": threshold_f,
                                "mean_net_return": mean_f,
                                "hit_rate": hit_f,
                                "n_trades": n_i,
                                "source": str(policy_path),
                            }
                        )
                deployed = deployed_rows.get(sid)
                if deployed is not None:
                    rows.append(deployed)
                if not rows:
                    continue
                frame = pd.DataFrame(rows).sort_values("threshold")
                for alias in strategy_rank_reference_aliases(sid, side):
                    if alias:
                        out[str(alias)] = frame
        except Exception:
            out = {}
        self._strategy_ev_threshold_tables = out
        return self._strategy_ev_threshold_tables

    def strategy_threshold_for_ev(
        self,
        *,
        strategy_id: str,
        side: str | None = None,
        target_mean_net_return: float,
        min_hit_rate: float = 0.60,
        fallback_threshold: float = 1.0,
    ) -> AuctionEvThresholdResult:
        """Return the lowest per-strategy rank floor meeting EV and hit-rate constraints."""
        tables = self._load_strategy_ev_threshold_tables()
        table = None
        for alias in strategy_rank_reference_aliases(strategy_id, side):
            table = tables.get(alias)
            if table is not None:
                break
        target = float(target_mean_net_return)
        hit_target = float(min_hit_rate)
        if table is None or table.empty:
            return AuctionEvThresholdResult(
                threshold=float(fallback_threshold),
                target_mean_net_return=target,
                target_hit_rate=hit_target,
                mean_net_return=float("nan"),
                hit_rate=float("nan"),
                n_trades=0,
                source="",
                enabled=False,
                reason="missing_strategy_ev_threshold_table",
            )
        ok = table[
            (pd.to_numeric(table["mean_net_return"], errors="coerce") >= target)
            & (pd.to_numeric(table["hit_rate"], errors="coerce") >= hit_target)
        ]
        if ok.empty:
            best = table.sort_values(
                ["mean_net_return", "hit_rate", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
            return AuctionEvThresholdResult(
                threshold=float(fallback_threshold),
                target_mean_net_return=target,
                target_hit_rate=hit_target,
                mean_net_return=float(best.get("mean_net_return", np.nan)),
                hit_rate=float(best.get("hit_rate", np.nan)),
                n_trades=int(best.get("n_trades", 0)),
                source=str(best.get("source", "")),
                enabled=False,
                reason="no_strategy_threshold_meets_ev_and_hit_rate_constraints",
            )
        row = ok.sort_values("threshold").iloc[0]
        return AuctionEvThresholdResult(
            threshold=float(row["threshold"]),
            target_mean_net_return=target,
            target_hit_rate=hit_target,
            mean_net_return=float(row["mean_net_return"]),
            hit_rate=float(row["hit_rate"]),
            n_trades=int(row["n_trades"]),
            source=str(row.get("source", "")),
            enabled=True,
            reason="strategy_ev_threshold",
        )

    def _load_strategy_ev_gate_table(self) -> dict[str, dict[str, Any]]:
        if self._strategy_ev_gate_table is not None:
            return self._strategy_ev_gate_table
        path = (
            self.data_root
            / "artifacts"
            / self.run_id
            / "policy_params"
            / "best_policy_params.json"
        )
        out: dict[str, dict[str, Any]] = {}
        try:
            params = json.loads(path.read_text(encoding="utf-8"))
            for strategy in params.get("strategies") or []:
                sid = str(strategy.get("strategy_id") or "")
                if not sid:
                    continue
                try:
                    mean_net = float(strategy.get("avg_net_pnl_per_trade"))
                    hit_rate = float(
                        strategy.get(
                            "pnl_positive_rate",
                            strategy.get("hit_rate"),
                        )
                    )
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(mean_net) and np.isfinite(hit_rate)):
                    continue
                row = {
                    "strategy_id": sid,
                    "mean_net_return": mean_net,
                    "hit_rate": hit_rate,
                    "source": str(path),
                }
                aliases = strategy_rank_reference_aliases(
                    sid,
                    str(strategy.get("side") or "").strip().lower() or None,
                )
                aliases.append(strategy_core_id(sid))
                for alias in aliases:
                    if alias:
                        out[str(alias)] = row
            self._strategy_ev_gate_table = out
        except Exception:
            self._strategy_ev_gate_table = {}
        return self._strategy_ev_gate_table

    def strategy_ev_gate(
        self,
        *,
        strategy_id: str,
        side: str | None = None,
        target_mean_net_return: float,
        min_hit_rate: float = 0.60,
    ) -> StrategyEvGateResult:
        table = self._load_strategy_ev_gate_table()
        row = None
        for alias in strategy_rank_reference_aliases(strategy_id, side):
            row = table.get(alias)
            if row is not None:
                break
        target = float(target_mean_net_return)
        hit_target = float(min_hit_rate)
        if row is None:
            return StrategyEvGateResult(
                allowed=False,
                target_mean_net_return=target,
                min_hit_rate=hit_target,
                mean_net_return=float("nan"),
                hit_rate=float("nan"),
                source="",
                reason="missing_strategy_policy_ev_metrics",
            )
        mean_net = float(row.get("mean_net_return", np.nan))
        hit_rate = float(row.get("hit_rate", np.nan))
        allowed = (
            np.isfinite(mean_net)
            and np.isfinite(hit_rate)
            and mean_net >= target
            and hit_rate >= hit_target
        )
        return StrategyEvGateResult(
            allowed=bool(allowed),
            target_mean_net_return=target,
            min_hit_rate=hit_target,
            mean_net_return=mean_net,
            hit_rate=hit_rate,
            source=str(row.get("source", "")),
            reason="strategy_ev_gate_pass" if allowed else "strategy_ev_gate_failed",
        )


def apply_policy_rank_percentile_gate(
    decision: Dict[str, Any],
    *,
    store: PolicyRankReferenceStore | None,
    allow_live_batch_rank_fallback_for_debug: bool = False,
    inference_min_base_train_rank_pct: float | None = None,
    require_cross_strategy_auction_rank: bool = False,
    use_auction_rank_for_threshold: bool = True,
) -> tuple[bool, str | None]:
    """Populate and enforce the live rank-percentile gate for one decision row."""
    threshold_space = str(decision.get("threshold_space") or "rank_percentile")
    if threshold_space != "rank_percentile":
        return True, None

    chain = dict(decision.get("chain_results") or {})
    result = (
        store.lookup(
            strategy_id=str(decision.get("strategy_id") or ""),
            side=str(decision.get("side") or ""),
            calibrated_score=float(decision.get("calibrated_score", np.nan)),
        )
        if store is not None
        else PolicyRankLookupResult(float("nan"), 0, "", "")
    )
    if np.isfinite(result.policy_rank_pct):
        policy_rank_pct = float(np.clip(result.policy_rank_pct, 0.0, 1.0))
        rank_source = "policy_rank_reference_percentile"
    elif allow_live_batch_rank_fallback_for_debug:
        policy_rank_pct = float(decision.get("sizer_rank_percentile", np.nan))
        rank_source = "live_batch_percentile_fallback_debug"
    else:
        decision["policy_rank_pct"] = np.nan
        decision["policy_rank_reference_n"] = int(result.n_rows)
        decision["policy_rank_reference_source"] = result.source
        decision["rank_score_source"] = "missing_policy_rank_reference_percentile"
        chain.update(
            {
                "policy_rank_pct": np.nan,
                "policy_rank_reference_n": int(result.n_rows),
                "policy_rank_reference_source": result.source,
                "rank_score_source": "missing_policy_rank_reference_percentile",
            }
        )
        decision["chain_results"] = chain
        return False, "missing_policy_rank_reference_percentile"

    decision["policy_rank_pct"] = policy_rank_pct
    decision["policy_rank_reference_n"] = int(result.n_rows)
    decision["policy_rank_reference_source"] = result.source
    decision["rank_score_source"] = rank_source
    decision["rank_percentile"] = policy_rank_pct
    decision["sizer_rank_percentile"] = policy_rank_pct
    decision["threshold_score"] = policy_rank_pct
    chain.update(
        {
            "policy_rank_pct": policy_rank_pct,
            "policy_rank_reference_n": int(result.n_rows),
            "policy_rank_reference_source": result.source,
            "rank_score_source": rank_source,
            "rank_percentile": policy_rank_pct,
            "sizer_rank_percentile": policy_rank_pct,
        }
    )

    auction = (
        store.lookup_auction(
            calibrated_score=float(decision.get("calibrated_score", np.nan))
        )
        if store is not None
        else PolicyRankLookupResult(float("nan"), 0, "", "cross_strategy")
    )
    threshold_rank_pct = policy_rank_pct
    threshold_rank_source = rank_source
    if np.isfinite(auction.policy_rank_pct):
        auction_rank_pct = float(np.clip(auction.policy_rank_pct, 0.0, 1.0))
        decision["normalized_rank_score"] = auction_rank_pct
        decision["auction_rank_pct"] = auction_rank_pct
        decision["auction_rank_reference_n"] = int(auction.n_rows)
        decision["auction_rank_reference_source"] = auction.source
        decision["auction_rank_score_source"] = "cross_strategy_auction_reference"
        if use_auction_rank_for_threshold:
            decision["threshold_score"] = auction_rank_pct
            threshold_rank_pct = auction_rank_pct
            threshold_rank_source = "cross_strategy_auction_reference"
        chain.update(
            {
                "normalized_rank_score": auction_rank_pct,
                "auction_rank_pct": auction_rank_pct,
                "auction_rank_reference_n": int(auction.n_rows),
                "auction_rank_reference_source": auction.source,
                "auction_rank_score_source": "cross_strategy_auction_reference",
            }
        )
    else:
        decision["normalized_rank_score"] = policy_rank_pct
        decision["auction_rank_pct"] = np.nan
        decision["auction_rank_reference_n"] = int(auction.n_rows)
        decision["auction_rank_reference_source"] = auction.source
        decision["auction_rank_score_source"] = (
            "missing_cross_strategy_auction_reference"
        )
        threshold_rank_source = "policy_rank_reference_percentile"
        chain.update(
            {
                "normalized_rank_score": policy_rank_pct,
                "auction_rank_pct": np.nan,
                "auction_rank_reference_n": int(auction.n_rows),
                "auction_rank_reference_source": auction.source,
                "auction_rank_score_source": "missing_cross_strategy_auction_reference",
            }
        )
        if require_cross_strategy_auction_rank:
            decision["chain_results"] = chain
            return False, "missing_cross_strategy_auction_reference"

    floor = inference_min_base_train_rank_pct
    if floor is not None:
        try:
            base_rank = float(
                chain.get(
                    "base_train_rank_pct",
                    decision.get("base_train_rank_pct", np.nan),
                )
            )
            floor_f = float(floor)
        except (TypeError, ValueError):
            base_rank = float("nan")
            floor_f = float("nan")
        if np.isfinite(base_rank) and np.isfinite(floor_f) and base_rank < floor_f:
            decision["chain_results"] = chain
            return False, "base_train_rank_below_floor"

    threshold = float(decision.get("effective_threshold", 1.0))
    decision["threshold_rank_score"] = threshold_rank_pct
    decision["threshold_rank_score_source"] = threshold_rank_source
    chain["threshold_rank_score"] = threshold_rank_pct
    chain["threshold_rank_score_source"] = threshold_rank_source
    decision["chain_results"] = chain
    if not np.isfinite(threshold_rank_pct) or threshold_rank_pct < threshold:
        return False, "rank_below_dynamic_threshold"
    return True, None
