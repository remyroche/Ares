import re

with open("extreme_price_movements/labeling.py", "r") as f:
    c = f.read()

target = """    for asset, lbs_or_out, rets, qual, conflict_j, tp_arr, sl_arr in results:"""
replace = """    out_mfe = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_mae = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_t_mfe = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
    out_t_mae = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)

    for asset, lbs_or_out, rets, qual, conflict_j, tp_arr, sl_arr, mfe_arr, mae_arr, t_mfe, t_mae in results:"""
c = c.replace(target, replace)

target2 = """        out_labels[asset] = lbs_or_out
        out_returns[asset] = rets
        if return_outcomes and qual is not None:
            out_quality[asset] = qual

    if return_outcomes:
        return out_labels, out_returns, out_quality
    return out_labels, out_returns"""
replace2 = """        out_labels[asset] = lbs_or_out
        out_returns[asset] = rets
        out_mfe[asset] = mfe_arr
        out_mae[asset] = mae_arr
        out_t_mfe[asset] = t_mfe
        out_t_mae[asset] = t_mae
        if return_outcomes and qual is not None:
            out_quality[asset] = qual

    if return_outcomes:
        return out_labels, out_returns, out_quality, out_mfe, out_mae, out_t_mfe, out_t_mae
    return out_labels, out_returns, out_mfe, out_mae, out_t_mfe, out_t_mae"""
c = c.replace(target2, replace2)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(c)
