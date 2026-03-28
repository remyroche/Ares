import re

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "r") as f:
    c = f.read()

# Insert the MFE/MAE filtering after Ridge model evaluation
target = """    pred = np.full(len(events), 0.5, dtype=np.float32)
    pred[valid_mask] = pred_event
    decile_spread = oof_payoff_decile_spread(pred_event, payoff_event)"""

replace = """    pred = np.full(len(events), 0.5, dtype=np.float32)
    pred[valid_mask] = pred_event
    decile_spread = oof_payoff_decile_spread(pred_event, payoff_event)

    # ── MFE/MAE Geometry Filtering ──
    _mfe = events["mfe"].to_numpy(dtype=float)
    _mae = events["mae"].to_numpy(dtype=float)
    _t_mfe = events["t_mfe"].to_numpy(dtype=float)
    _t_mae = events["t_mae"].to_numpy(dtype=float)

    _mfe_mae_ratio = np.clip(_mfe / (_mae + EPS), 0.0, 12.0)
    _mfe_before_mae = (_t_mfe < _t_mae).astype(float)

    median_mfe_mae = float(np.nanmedian(_mfe_mae_ratio)) if len(_mfe_mae_ratio) > 0 else 0.0
    p10_mfe_mae = float(np.nanpercentile(_mfe_mae_ratio, 10)) if len(_mfe_mae_ratio) > 0 else 0.0
    p90_mae = float(np.nanpercentile(_mae, 90)) if len(_mae) > 0 else 0.0
    p50_mfe = float(np.nanpercentile(_mfe, 50)) if len(_mfe) > 0 else 0.0
    pct_mfe_before_mae = float(np.nanmean(_mfe_before_mae)) if len(_mfe_before_mae) > 0 else 0.0

    if median_mfe_mae >= 1.5 or p10_mfe_mae >= 0.45 or p90_mae <= 0.65 * p50_mfe or pct_mfe_before_mae >= 0.56:
        tprint(
            f"[eval:{cfg_id}] EARLY_EXIT geometric filter hit: "
            f"median_mfe_mae={median_mfe_mae:.2f} (fail>=1.5) "
            f"p10_mfe_mae={p10_mfe_mae:.2f} (fail>=0.45) "
            f"p90_mae={p90_mae:.4f} vs 0.65*p50_mfe={0.65*p50_mfe:.4f} "
            f"pct_mfe_before_mae={pct_mfe_before_mae:.3f} (fail>=0.56)"
        )
        return _empty_result(
            cfg,
            cfg_id,
            full_n,
            reason="failed_geometric_mfe_mae_filter",
            stage2_rescore=stage2_rescore,
        )"""

c = c.replace(target, replace)

with open("extreme_price_movements/offline_optimisers/compare_tbm_parameters.py", "w") as f:
    f.write(c)
