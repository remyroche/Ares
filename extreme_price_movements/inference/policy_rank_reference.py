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
    manifest["strategies"] = strategies
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


@dataclass(frozen=True)
class PolicyRankLookupResult:
    policy_rank_pct: float
    n_rows: int
    source: str
    strategy_id: str


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


def apply_policy_rank_percentile_gate(
    decision: Dict[str, Any],
    *,
    store: PolicyRankReferenceStore | None,
    allow_live_batch_rank_fallback_for_debug: bool = False,
    inference_min_base_train_rank_pct: float | None = None,
    require_cross_strategy_auction_rank: bool = False,
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
