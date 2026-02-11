import bisect
from math import erf, isnan
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set, Union

THRESHOLDS_ALL   = [25, 50, 66, 75, 85, 90]
RARE_CANDIDATES  = [85, 90]
BROAD_CANDIDATES = [50, 66]

GLOBAL_P_MIN = 0.005   # 0.5%
GLOBAL_P_MAX = 0.995   # 99.5%

EPS = 1e-12

# z targets used only for tie-breaks if skill is similar / missing
Z_TARGET_RARE  = -1.25
Z_TARGET_BROAD = +0.50

LAMBDA_Z = 0.03        # strength of z-penalty in selection score (small)


def _normal_cdf(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    arr = x.to_numpy(dtype=np.float64) / np.sqrt(2.0)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(arr))
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(cdf, index=x.index, columns=x.columns, dtype=np.float32)
    return pd.Series(cdf, index=x.index, dtype=np.float32)


def _rolling_percentile_exact(s: pd.Series, n: int) -> pd.Series:
    vals = s.to_numpy(dtype=np.float64)
    out = np.full(len(vals), 0.5, dtype=np.float32)
    window = []
    for i, cur in enumerate(vals):
        if i > 0:
            prev = vals[i - 1]
            bisect.insort(window, prev)
            if len(window) > n:
                old = vals[i - 1 - n]
                j = bisect.bisect_left(window, old)
                if j < len(window):
                    window.pop(j)
        if len(window) < n or not np.isfinite(cur):
            continue
        rank = bisect.bisect_right(window, cur)
        out[i] = rank / float(n)
    return pd.Series(out, index=s.index, dtype=np.float32)


def add_gate_features(
    df: pd.DataFrame,
    s_col: str,
    prefix: str,
    n: int = 256,
    add_strict: bool = True,
    percentile_mode: str = "approx",
    min_std: float = 1e-6,
) -> pd.DataFrame:
    """Original (Series based / Single column) implementation. Now deprecated for Panel use."""
    if s_col not in df.columns:
        raise KeyError(f"Missing score column: {s_col}")

    s = df[s_col].astype(np.float32)

    roll = s.rolling(n, min_periods=n)
    mean = roll.mean().shift(1)
    std = roll.std(ddof=0).shift(1).clip(lower=min_std)
    z = ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    if percentile_mode == "exact":
        pct = _rolling_percentile_exact(s, n)
    else:
        pct = _normal_cdf(z).clip(0.0, 1.0).fillna(0.5).astype(np.float32)

    bins = np.digitize(pct.to_numpy(), bins=[1.0 / 3.0, 2.0 / 3.0]).astype(np.int8)

    df[f"{prefix}_mean_{n}"] = mean.fillna(0.0).astype(np.float32)
    df[f"{prefix}_std_{n}"] = std.fillna(min_std).astype(np.float32)
    df[f"{prefix}_z_{n}"] = z
    df[f"{prefix}_pct_{n}"] = pct.astype(np.float32)
    df[f"{prefix}_bin3_{n}"] = bins

    if add_strict:
        df[f"{prefix}_gt25_{n}"] = (pct > 0.25).astype(np.int8)
        df[f"{prefix}_gt50_{n}"] = (pct > 0.50).astype(np.int8)
        df[f"{prefix}_gt66_{n}"] = (pct > 0.66).astype(np.int8)
        df[f"{prefix}_gt75_{n}"] = (pct > 0.75).astype(np.int8)
        df[f"{prefix}_gt85_{n}"] = (pct > 0.85).astype(np.int8)
        df[f"{prefix}_gt90_{n}"] = (pct > 0.90).astype(np.int8)

    return df


def add_gate_features_panel(
    panel_s: pd.DataFrame, # (Time, Assets)
    prefix: str,
    n: int = 256,
    add_strict: bool = True,
    percentile_mode: str = "approx",
    min_std: float = 1e-6,
) -> Dict[str, pd.DataFrame]:
    """
    Panel-aware version of add_gate_features.
    Computes rolling stats per asset (column-wise). Returns dict of feature_names -> Panel DataFrame.
    """
    s = panel_s.astype(np.float32)

    # Rolling per column (asset)
    roll = s.rolling(n, min_periods=n)
    mean = roll.mean().shift(1)
    std = roll.std(ddof=0).shift(1).clip(lower=min_std)
    # Z-score normalization per asset history
    z = ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    if percentile_mode == "exact":
        # Slow for panel, iterate columns? Or ensure _rolling_percentile_exact handles 2D?
        # Numba optimization recommended for exact percentile on large panels.
        # Fallback to approx for now or iterate.
        # Approx is standard for Gates.
        pct = _normal_cdf(z).clip(0.0, 1.0).fillna(0.5).astype(np.float32)
    else:
        pct = _normal_cdf(z).clip(0.0, 1.0).fillna(0.5).astype(np.float32)

    bins = np.digitize(pct.to_numpy(), bins=[1.0 / 3.0, 2.0 / 3.0]).astype(np.int8)
    bins_df = pd.DataFrame(bins, index=s.index, columns=s.columns)

    out = {}
    out[f"{prefix}_mean_{n}"] = mean.fillna(0.0).astype(np.float32)
    out[f"{prefix}_std_{n}"] = std.fillna(min_std).astype(np.float32)
    out[f"{prefix}_z_{n}"] = z
    out[f"{prefix}_pct_{n}"] = pct.astype(np.float32)
    out[f"{prefix}_bin3_{n}"] = bins_df

    if add_strict:
        out[f"{prefix}_gt25_{n}"] = (pct > 0.25).astype(np.int8)
        out[f"{prefix}_gt50_{n}"] = (pct > 0.50).astype(np.int8)
        out[f"{prefix}_gt66_{n}"] = (pct > 0.66).astype(np.int8)
        out[f"{prefix}_gt75_{n}"] = (pct > 0.75).astype(np.int8)
        out[f"{prefix}_gt85_{n}"] = (pct > 0.85).astype(np.int8)
        out[f"{prefix}_gt90_{n}"] = (pct > 0.90).astype(np.int8)

    return out


def add_accept_gate_features(
    df: pd.DataFrame,
    s_col: str = "accept_score",
    N: int = 256,
    add_strict: bool = True,
    percentile_mode: str = "approx",
) -> pd.DataFrame:
    return add_gate_features(
        df=df,
        s_col=s_col,
        prefix="s",
        n=N,
        add_strict=add_strict,
        percentile_mode=percentile_mode,
    )


# --------- Feature Selection Helpers (User Request) ---------

def prevalence(binary_gate_col: Union[pd.Series, pd.DataFrame], train_mask: pd.Series) -> float:
    # p = mean(gate==1) over training rows only
    # Handle Panel DataFrame by flattening valid values
    if isinstance(binary_gate_col, pd.DataFrame):
        # Slice time index using train_mask
        subset = binary_gate_col.loc[train_mask.index[train_mask]]
        total_elements = subset.size
        # Count ones
        ones_count = (subset == 1).sum().sum()
        return float(ones_count / (total_elements + EPS)) if total_elements > 0 else 0.0
    
    subset = binary_gate_col[train_mask]
    if len(subset) == 0:
        return 0.0
    return float((subset == 1).mean())

def within_family_zscores(p_by_thr: Dict[int, float]) -> Tuple[Dict[int, float], float, float]:
    # p_by_thr: dict thr -> p
    ps = [p_by_thr[t] for t in THRESHOLDS_ALL if t in p_by_thr]
    if not ps:
        return {}, 0.0, 0.0
    mu = float(np.mean(ps))
    sd = float(np.std(ps))  # across thresholds within family
    z = {}
    for t, p in p_by_thr.items():
        z[t] = (p - mu) / (sd + EPS)
    return z, mu, sd

def robust_skill_metric(gate_col: Union[pd.Series, pd.DataFrame], target: Union[pd.Series, pd.DataFrame], time_blocks: Optional[List[pd.Series]], train_mask: Optional[pd.Series]) -> Optional[float]:
    # "Cheap" but stable: block-median IC of gate vs target
    # Memory-optimized: uses column-wise correlation instead of .stack()
    if target is None:
        return None

    scores = []
    
    is_panel = isinstance(gate_col, pd.DataFrame)
    
    # If no blocks provided, treat whole range as one block
    if time_blocks is None:
        time_blocks = [pd.Series(True, index=gate_col.index)]
    
    for block_mask in time_blocks:
        if train_mask is None:
             tm = block_mask
        else:
             tm = (train_mask & block_mask)
             
        if not tm.any():
             continue
             
        if is_panel:
            g_sub = gate_col.loc[tm]
            if isinstance(target, pd.DataFrame):
                t_sub = target.loc[tm]
            else:
                t_sub = target.loc[tm]
            
            # Memory-efficient: compute per-column Spearman, then average
            # Avoids .stack() which creates N_rows × N_cols elements
            col_scores = []
            shared_cols = g_sub.columns.intersection(t_sub.columns) if isinstance(target, pd.DataFrame) else g_sub.columns
            for col in shared_cols:
                g_vec = g_sub[col].dropna()
                if isinstance(target, pd.DataFrame):
                    t_vec = t_sub[col].reindex(g_vec.index).dropna()
                else:
                    t_vec = t_sub.reindex(g_vec.index).dropna()
                common = g_vec.index.intersection(t_vec.index)
                if len(common) < 20:
                    continue
                c = g_vec.loc[common].corr(t_vec.loc[common], method="spearman")
                if not isnan(c):
                    col_scores.append(c)
            if len(col_scores) >= 3:
                score = float(np.median(col_scores))
            else:
                continue
        else:
            m = tm
            if m.sum() < 50:
                continue
            scan_subset = gate_col[m]
            target_subset = target[m]
            score = scan_subset.corr(target_subset, method="spearman")
            
        if not isnan(score):
            scores.append(score)
            
    if len(scores) == 0:
        return None
    return float(np.median(scores))

def select_with_score(candidates: List[int], gate_cols: Dict[int, Union[pd.Series, pd.DataFrame]], skill_by_thr: Dict[int, Optional[float]], z_by_thr: Dict[int, float], z_target: float) -> Optional[int]:
    # Returns best threshold among candidates.
    # Score = skill - λ*|z - z_target|; if skill missing, use only z distance.
    best_t = None
    best_s = -float('inf')
    
    for t in candidates:
        if t not in gate_cols:
            continue
        skill = skill_by_thr.get(t, None)
        z = z_by_thr.get(t, 0.0)

        if skill is None:
            # no skill available => purely choose closest to target z
            score = -abs(z - z_target)
        else:
            score = skill - LAMBDA_Z * abs(z - z_target)

        if score > best_s:
            best_s = score
            best_t = t
    return best_t


def select_gated_features(
    gate_feature_table: Dict[str, Union[pd.Series, pd.DataFrame]],     # Dict: feature_name -> data container
    families: List[Tuple[str, int]],      # iterable of (interaction_name, window) families
    target: Optional[pd.Series] = None,   # numeric vector aligned with rows
    time_blocks: Optional[List[pd.Series]] = None, # list of boolean masks
    train_mask: Optional[pd.Series] = None  # boolean mask for train rows
) -> List[str]:
    
    if train_mask is None:
        # Default to all True if no mask provided. Use index from first item in table.
        first_key = next(iter(gate_feature_table))
        first_item = gate_feature_table[first_key]
        train_mask = pd.Series(True, index=first_item.index)
        
    selected = []

    for (interaction, window) in families:
        # Build the family feature-name map: threshold -> column
        gate_cols = {}  # thr -> Series/DataFrame
        p_by_thr   = {}

        for t in THRESHOLDS_ALL:
            # Match naming from add_gate_features_panel: {prefix}_gt{t}_{n}
            feat_name = f"{interaction}_gt{t}_{window}"
            if feat_name not in gate_feature_table:
                continue
            col = gate_feature_table[feat_name]
            gate_cols[t] = col
            p_by_thr[t]  = prevalence(col, train_mask)

        # If family incomplete or empty, skip
        if len(gate_cols) == 0:
            continue

        # 1) Global sanity filter: drop too-rare/too-common gates
        gate_cols_f = {}
        p_by_thr_f  = {}
        for t, col in gate_cols.items():
            p = p_by_thr[t]
            if (p >= GLOBAL_P_MIN) and (p <= GLOBAL_P_MAX):
                gate_cols_f[t] = col
                p_by_thr_f[t]  = p

        # If everything got filtered out, keep nothing or fallback?
        # User said "fallback: choose gt85 + gt50 if present"
        if len(gate_cols_f) == 0:
            # fallback: choose gt85 + gt50 if present
            if 85 in gate_cols: selected.append(f"{interaction}_gt85_{window}")
            if 50 in gate_cols: selected.append(f"{interaction}_gt50_{window}")
            continue

        # 2) Compute within-family relative prevalence z-scores
        z_by_thr, mu_F, sd_F = within_family_zscores(p_by_thr_f)

        # 3) Optional skill metric per gate (block-robust)
        skill_by_thr = {}
        if target is not None and time_blocks is not None:
             for t, col in gate_cols_f.items():
                skill_by_thr[t] = robust_skill_metric(col, target, time_blocks, train_mask)
        else:
             # No skill metric available
             for t in gate_cols_f.keys():
                 skill_by_thr[t] = None

        # 4) Pick ONE rare gate and ONE broad gate (max 2 thresholds per family)
        rare_thr  = select_with_score(
            candidates=[t for t in RARE_CANDIDATES if t in gate_cols_f],
            gate_cols=gate_cols_f,
            skill_by_thr=skill_by_thr,
            z_by_thr=z_by_thr,
            z_target=Z_TARGET_RARE
        )

        broad_thr = select_with_score(
            candidates=[t for t in BROAD_CANDIDATES if t in gate_cols_f],
            gate_cols=gate_cols_f,
            skill_by_thr=skill_by_thr,
            z_by_thr=z_by_thr,
            z_target=Z_TARGET_BROAD
        )

        # 5) Ensure distinct thresholds; if collision / missing, backfill with next-best
        chosen = set()
        if rare_thr is not None:
            chosen.add(rare_thr)
        if broad_thr is not None:
            chosen.add(broad_thr)

        # If only one chosen, fill with next best from remaining thresholds using same score logic
        if len(chosen) < 2:
            remaining = [t for t in gate_cols_f.keys() if t not in chosen]
            # pick best remaining by skill - λ*|z| (no target) as a generic backfill
            best_t = None
            best_s = -float('inf')
            for t in remaining:
                skill = skill_by_thr.get(t, None)
                z = z_by_thr.get(t, 0.0)
                # If skill missing, use relative z-score magnitude as penalty (prefer closer to mean?)
                # Or just prioritize largest Z-score?
                # User code: score = (-abs(z)) if skill is None else (skill - LAMBDA_Z * abs(z))
                # Wait, "no target" for generic backfill => maximize score (closest to mean Z=0?)
                # Yes, -abs(z) means closer to mean prevalence within family.
                score = (-abs(z)) if skill is None else (skill - LAMBDA_Z * abs(z))
                if score > best_s:
                    best_s, best_t = score, t
            if best_t is not None:
                chosen.add(best_t)

        # 6) Add selected feature names
        for t in sorted(chosen):
            selected.append(f"{interaction}_gt{t}_{window}")

    return selected
