from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from extreme_price_movements.src_utils_tprint import tprint
except Exception:
    tprint = print

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _rolling_percentile_nb(
        scores: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        window: int,
    ) -> np.ndarray:
        n = scores.shape[0]
        out = np.full(n, np.nan, dtype=np.float32)
        for g in range(starts.shape[0]):
            lo = starts[g]
            hi = ends[g]
            for i in range(lo, hi + 1):
                if not np.isfinite(scores[i]):
                    continue
                start = max(lo, i - window)
                count = 0
                below = 0
                for j in range(start, i):
                    if np.isfinite(scores[j]):
                        count += 1
                        if scores[j] <= scores[i]:
                            below += 1
                if count > 0:
                    out[i] = np.float32(below) / np.float32(count)
        return out

    def rolling_asset_percentile(
        scores: np.ndarray,
        symbols: Iterable[Any],
        timestamps: Iterable[Any] | None = None,
        window: int = 240,
    ) -> np.ndarray:
        vals = np.asarray(scores, dtype=np.float32)
        n = len(vals)
        tprint(f"[trade_filtering] rolling_asset_percentile start: n={n} window={int(window)}")
        if n == 0:
            return np.full(n, np.nan, dtype=np.float32)
        sym = np.asarray(list(symbols))
        _, group_ids = np.unique(sym, return_inverse=True)
        group_ids = group_ids.astype(np.int32)
        n_groups = int(group_ids.max()) + 1
        starts = np.zeros(n_groups, dtype=np.int64)
        ends = np.zeros(n_groups, dtype=np.int64)
        for g in range(n_groups):
            mask = group_ids == g
            idxs = np.where(mask)[0]
            starts[g] = idxs[0]
            ends[g] = idxs[-1]
        w = max(int(window), 5)
        if timestamps is not None:
            ts = np.asarray(pd.to_datetime(list(timestamps), errors="coerce").astype(np.int64))
            order = np.lexsort((group_ids, ts))
            vals_s = vals[order].copy()
            new_gids = group_ids[order]
            starts_s = np.zeros(n_groups, dtype=np.int64)
            ends_s = np.zeros(n_groups, dtype=np.int64)
            for g in range(n_groups):
                mask = new_gids == g
                idxs = np.where(mask)[0]
                starts_s[g] = idxs[0]
                ends_s[g] = idxs[-1]
            ranks_s = _rolling_percentile_nb(vals_s, starts_s, ends_s, w)
            out = np.full(n, np.nan, dtype=np.float32)
            out[order] = ranks_s
        else:
            out = _rolling_percentile_nb(vals, starts, ends, w)
        tprint(
            f"[trade_filtering] rolling_asset_percentile done: finite={int(np.isfinite(out).sum())}/{n}"
        )
        return np.clip(out, 0.0, 1.0)

else:

    def rolling_asset_percentile(
        scores: np.ndarray,
        symbols: Iterable[Any],
        timestamps: Iterable[Any] | None = None,
        window: int = 240,
    ) -> np.ndarray:
        vals = np.asarray(scores, dtype=np.float32)
        n = len(vals)
        tprint(
            f"[trade_filtering] rolling_asset_percentile start: n={n} window={int(window)}"
        )
        out = np.full(n, np.nan, dtype=np.float32)
        if n == 0:
            return out
        sym = np.asarray(list(symbols))
        ts = (
            pd.to_datetime(list(timestamps), errors="coerce")
            if timestamps is not None
            else None
        )
        df = pd.DataFrame({"score": vals, "symbol": sym})
        if ts is not None:
            df["ts"] = ts
            df = df.sort_values(["symbol", "ts"]).copy()
        else:
            df = df.copy()
        df["orig_idx"] = np.arange(len(df), dtype=np.int64)
        ranks = np.full(len(df), np.nan, dtype=np.float32)
        for _, g in df.groupby("symbol", sort=False):
            gv = g["score"].to_numpy(dtype=np.float32)
            idx = g.index.to_numpy(dtype=np.int64)
            for i in range(len(gv)):
                lo = max(0, i - int(window))
                hist = gv[lo:i]
                if hist.size == 0 or not np.isfinite(gv[i]):
                    ranks[idx[i]] = np.nan
                    continue
                finite = hist[np.isfinite(hist)]
                if finite.size == 0:
                    ranks[idx[i]] = np.nan
                    continue
                ranks[idx[i]] = float(np.mean(finite <= gv[i]))
        if ts is not None:
            back = pd.Series(ranks, index=df["orig_idx"].to_numpy(dtype=np.int64))
            out = back.reindex(np.arange(n)).to_numpy(dtype=np.float32)
        else:
            out = ranks.astype(np.float32)
        tprint(
            f"[trade_filtering] rolling_asset_percentile done: finite={int(np.isfinite(out).sum())}/{n}"
        )
        return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class TopRankMaskResult:
    mask: np.ndarray
    chosen_topx: int
    coverage: float
    score: float


def get_or_select_top_rank_mask(
    *,
    strategy_id: str,
    cache: dict[str, TopRankMaskResult] | None = None,
    **kwargs,
) -> TopRankMaskResult:
    sid = str(strategy_id or "").strip()
    if cache is not None and sid and sid in cache:
        cached = cache[sid]
        tprint(
            f"[trade_filtering] reusing cached mask for strategy_id={sid[:80]} "
            f"topx={cached.chosen_topx} kept={int(cached.mask.sum())}/{len(cached.mask)}"
        )
        return cached

    result = select_top_rank_mask(**kwargs)
    if cache is not None and sid:
        cache[sid] = result
        tprint(
            f"[trade_filtering] cached mask for strategy_id={sid[:80]} "
            f"topx={result.chosen_topx} kept={int(result.mask.sum())}/{len(result.mask)}"
        )
    return result


def select_top_rank_mask(
    *,
    base_prob: np.ndarray,
    strategy_mask: np.ndarray,
    symbols: Iterable[Any],
    timestamps: Iterable[Any] | None,
    outcomes: np.ndarray,
    mfe: np.ndarray | None,
    mae: np.ndarray | None,
    t_mfe: np.ndarray | None,
    t_mae: np.ndarray | None,
    tp: np.ndarray | None,
    topx_values: tuple[int, ...] = (20, 25, 30, 35, 40),
    rank_window: int = 240,
) -> TopRankMaskResult:
    """Choose top-x% rolling-rank mask maximizing lift*coverage*excess magnitude."""
    p = np.asarray(base_prob, dtype=np.float32)
    sm = np.asarray(strategy_mask, dtype=bool)
    y = np.asarray(outcomes, dtype=np.float32)
    n = len(p)
    if len(sm) != n:
        sm = np.ones(n, dtype=bool)
    if n == 0 or int(sm.sum()) == 0:
        tprint("[trade_filtering] select_top_rank_mask early-exit: empty input")
        return TopRankMaskResult(
            mask=np.zeros(n, dtype=bool),
            chosen_topx=int(topx_values[0]),
            coverage=0.0,
            score=0.0,
        )

    pct = rolling_asset_percentile(
        scores=p,
        symbols=symbols,
        timestamps=timestamps,
        window=max(int(rank_window), 5),
    )
    valid = sm & np.isfinite(pct)
    baseline_hit = float(np.mean(y[sm] >= 0.5)) if int(sm.sum()) else 0.0

    mfe_v = np.asarray(mfe, dtype=np.float32) if mfe is not None else None
    mae_v = np.asarray(mae, dtype=np.float32) if mae is not None else None
    tmfe_v = np.asarray(t_mfe, dtype=np.float32) if t_mfe is not None else None
    tmae_v = np.asarray(t_mae, dtype=np.float32) if t_mae is not None else None
    tp_v = np.asarray(tp, dtype=np.float32) if tp is not None else None

    best = (-1.0, int(topx_values[0]), np.zeros(n, dtype=bool), 0.0)
    for x in topx_values:
        thr = 1.0 - float(x) / 100.0
        m = valid & (pct >= thr)
        cov = float(np.mean(m))
        if cov <= 0.0:
            continue
        y_frac = float(np.clip(0.10 / max(cov, 1e-6), 0.01, 1.0))
        kept_idx = np.flatnonzero(m)
        if kept_idx.size == 0:
            continue
        scores_kept = p[kept_idx]
        n_top = max(1, int(np.ceil(y_frac * kept_idx.size)))
        top_idx = kept_idx[np.argsort(scores_kept)[-n_top:]]
        hit_top = float(np.mean(y[top_idx] >= 0.5)) if top_idx.size else 0.0
        lift = hit_top / max(baseline_hit, 1e-6)

        excess = 0.0
        if (
            mfe_v is not None
            and mae_v is not None
            and tmfe_v is not None
            and tmae_v is not None
            and tp_v is not None
        ):
            top_mfe = mfe_v[top_idx]
            top_mae = mae_v[top_idx]
            top_tmfe = tmfe_v[top_idx]
            top_tmae = tmae_v[top_idx]
            top_tp = np.clip(tp_v[top_idx], 1e-6, None)
            tp_before_sl = (top_mfe >= top_tp) & (
                (~np.isfinite(top_tmae)) | (top_tmfe <= top_tmae)
            )
            if tp_before_sl.any():
                excess = float(
                    np.mean(
                        np.maximum(top_mfe[tp_before_sl] - top_tp[tp_before_sl], 0.0)
                    )
                )

        score = float(lift * cov * max(excess, 1e-8))
        if score > best[0]:
            best = (score, int(x), m, cov)
        tprint(
            "[trade_filtering] candidate "
            f"topx={int(x)} cov={cov:.4f} y_frac={y_frac:.4f} lift={lift:.4f} "
            f"excess={excess:.6f} score={score:.6e}"
        )

    out = TopRankMaskResult(
        mask=best[2], chosen_topx=best[1], coverage=float(best[3]), score=float(best[0])
    )
    tprint(
        f"[trade_filtering] selected topx={out.chosen_topx} "
        f"coverage={out.coverage:.4f} score={out.score:.6e} kept={int(out.mask.sum())}/{n}"
    )
    return out
