from dataclasses import dataclass

import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint


@dataclass
class CompositeObjectiveConfig:
    mar: float = 0.0
    eps_log: float = 1e-12
    eps_sortino: float = 1e-12
    mode: str = "hard_gate"  # hard_gate | signed_product
    elg_scale: float = 10000.0
    mnpt_scale: float = 10000.0
    elg_min: float = -1.0
    elg_max: float = 1.0
    sortino_min: float = -10.0
    sortino_max: float = 10.0
    mnpt_min: float = -1.0
    mnpt_max: float = 1.0
    min_trades_per_fold: int = 200
    q_top: float = 0.95
    selection: str = "min_std"  # min_std | max_worst


def expected_log_growth(r_t, eps_log: float = 1e-12) -> float:
    tprint(f"Entering function: expected_log_growth in tp_sl_selection.py")
    r = np.asarray(r_t, dtype=float)
    if r.size == 0:
        return float("nan")
    r = np.clip(r, -1.0 + float(eps_log), None)
    return float(np.mean(np.log1p(r)))


def sortino_ratio(r_t, mar: float = 0.0, eps_sortino: float = 1e-12) -> float:
    tprint(f"Entering function: sortino_ratio in tp_sl_selection.py")
    r = np.asarray(r_t, dtype=float)
    if r.size == 0:
        return float("nan")
    ex = r - float(mar)
    downside = np.minimum(0.0, ex)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float(np.mean(ex) / (dd + float(eps_sortino)))


def mean_net_pnl_per_trade(pnl_net) -> float:
    tprint(f"Entering function: mean_net_pnl_per_trade in tp_sl_selection.py")
    p = np.asarray(pnl_net, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p))


def composite_objective(elg: float, sr: float, mnpt: float, cfg: CompositeObjectiveConfig) -> float:
    tprint(f"Entering function: composite_objective in tp_sl_selection.py")
    elg_s = np.clip(elg * float(cfg.elg_scale), cfg.elg_min, cfg.elg_max)
    sr_s = np.clip(sr, cfg.sortino_min, cfg.sortino_max)
    mnpt_s = np.clip(mnpt * float(cfg.mnpt_scale), cfg.mnpt_min, cfg.mnpt_max)

    if cfg.mode == "hard_gate":
        if (elg_s < 0.0) or (sr_s < 0.0) or (mnpt_s < 0.0):
            return float("-inf")

    sign = np.sign(elg_s) * np.sign(sr_s) * np.sign(mnpt_s)
    return float(sign * abs(elg_s) * abs(sr_s) * abs(mnpt_s))


def evaluate_fold_metrics(r_t, pnl_net, cfg: CompositeObjectiveConfig) -> dict:
    tprint(f"Entering function: evaluate_fold_metrics in tp_sl_selection.py")
    pnl = np.asarray(pnl_net, dtype=float)
    n_trades = int(len(pnl))
    elg = expected_log_growth(r_t, eps_log=cfg.eps_log)
    sr = sortino_ratio(r_t, mar=cfg.mar, eps_sortino=cfg.eps_sortino)
    mnpt = mean_net_pnl_per_trade(pnl)

    if n_trades < int(cfg.min_trades_per_fold):
        obj = float("-inf")
    else:
        obj = composite_objective(elg, sr, mnpt, cfg)

    return {
        "ELG": float(elg),
        "SR": float(sr),
        "MNPT": float(mnpt),
        "n_trades": n_trades,
        "Objective": float(obj),
    }


def aggregate_candidate_folds(candidate_to_fold_metrics: dict) -> pd.DataFrame:
    tprint(f"Entering function: aggregate_candidate_folds in tp_sl_selection.py")
    rows = []
    for cand, folds in candidate_to_fold_metrics.items():
        obj = np.array([f.get("Objective", np.nan) for f in folds], dtype=float)
        elg = np.array([f.get("ELG", np.nan) for f in folds], dtype=float)
        sr = np.array([f.get("SR", np.nan) for f in folds], dtype=float)
        mnpt = np.array([f.get("MNPT", np.nan) for f in folds], dtype=float)
        ntr = np.array([f.get("n_trades", np.nan) for f in folds], dtype=float)
        rows.append(
            {
                "candidate": cand,
                "Objective_mean": float(np.nanmean(obj)),
                "Objective_std": float(np.nanstd(obj)),
                "Objective_worst": float(np.nanmin(obj)),
                "ELG_mean": float(np.nanmean(elg)),
                "SR_mean": float(np.nanmean(sr)),
                "MNPT_mean": float(np.nanmean(mnpt)),
                "n_trades_mean": float(np.nanmean(ntr)),
                "fold_metrics": folds,
            }
        )
    return pd.DataFrame(rows)


def select_robust_default(summary_df: pd.DataFrame, cfg: CompositeObjectiveConfig) -> dict:
    tprint(f"Entering function: select_robust_default in tp_sl_selection.py")
    if summary_df.empty:
        raise ValueError("summary_df is empty")

    obj = np.asarray(summary_df["Objective_mean"].values, dtype=float)
    valid = np.isfinite(obj)
    if not np.any(valid):
        idx = int(np.argmax(summary_df["Objective_worst"].values))
        return summary_df.iloc[idx].to_dict()

    q = float(np.nanquantile(obj[valid], float(np.clip(cfg.q_top, 0.0, 1.0))))
    top = summary_df[summary_df["Objective_mean"] >= q].copy()
    if top.empty:
        top = summary_df.sort_values("Objective_mean", ascending=False).head(1)

    if cfg.selection == "max_worst":
        picked = top.sort_values(["Objective_worst", "Objective_std"], ascending=[False, True]).iloc[0]
    else:
        picked = top.sort_values(["Objective_std", "Objective_mean"], ascending=[True, False]).iloc[0]
    return picked.to_dict()


def build_tp_sl_grid(k_tp_grid, k_sl_grid) -> list[tuple[float, float]]:
    tprint(f"Entering function: build_tp_sl_grid in tp_sl_selection.py")
    return [(float(tp), float(sl)) for tp in k_tp_grid for sl in k_sl_grid]



def select_best_tp_sl(
    open_,
    close,
    event_idx,
    tp_mult_grid,
    sl_mult_grid,
    timestamps=None,
    cfg: CompositeObjectiveConfig | None = None,
):
    """Lightweight TP/SL selector using composite objective over candidate-adjusted returns.

    Returns dict with selected candidate and summary table.
    """
    tprint(f"Entering function: select_best_tp_sl in tp_sl_selection.py")
    cfg = cfg or CompositeObjectiveConfig()
    open_arr = np.asarray(open_, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    idx = np.asarray(event_idx, dtype=int)
    if idx.size == 0:
        return {"best": None, "summary": pd.DataFrame()}

    # Proxy event return (next-bar close-open where possible).
    i1 = np.clip(idx + 1, 0, len(open_arr) - 1)
    i2 = np.clip(idx + 2, 0, len(close_arr) - 1)
    base_ret = (close_arr[i2] - open_arr[i1]) / np.clip(np.abs(open_arr[i1]), 1e-12, None)

    if timestamps is None:
        ts = pd.Series(np.arange(len(base_ret)))
    else:
        ts = pd.to_datetime(np.asarray(timestamps)[idx], utc=True, errors="coerce")

    # Build deterministic fold ids after time order.
    order = np.argsort(ts.values.astype("datetime64[ns]") if hasattr(ts.values, 'dtype') else np.arange(len(ts)))
    fold_id = np.zeros(len(base_ret), dtype=int)
    k = min(3, max(1, len(base_ret) // 200))
    bins = np.array_split(order, k)
    for i, b in enumerate(bins):
        fold_id[b] = i

    cand_metrics = {}
    for k_tp, k_sl in build_tp_sl_grid(tp_mult_grid, sl_mult_grid):
        folds = []
        pnl_adj = np.where(base_ret >= 0.0, base_ret * float(k_tp), base_ret * float(k_sl))
        for fid in np.unique(fold_id):
            m = fold_id == fid
            grp = pd.DataFrame({"ts": ts[m], "p": pnl_adj[m]}).groupby("ts", as_index=False)["p"].sum()
            folds.append(evaluate_fold_metrics(r_t=grp["p"].values, pnl_net=pnl_adj[m], cfg=cfg))
        cand_metrics[(float(k_tp), float(k_sl))] = folds

    summary = aggregate_candidate_folds(cand_metrics)
    best = select_robust_default(summary, cfg)
    return {"best": best, "summary": summary}
