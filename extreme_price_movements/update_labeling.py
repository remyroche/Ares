import re

with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

search = """            if hit_sl and hit_tp:
                # Ambiguous bar - Assume SL
                outcomes[i] = OUT_SL
                returns[i] = -sl
                exit_idxs[i] = j
                # Loss Quality: did we see profit first?
                # f(MAE, MFE). Here we hit SL, so MAE >= SL_dist.
                # Quality depends on MFE seen before? We track MFE of *this* bar too.
                # MFE/TP_dist.
                den_tp = max(entry_p * abs(activation), _QUALITY_EPS)
                qual_raw = (mfe_val / den_tp) * 0.5
                qual = _soft_squash_pos(qual_raw)
                quality[i] = _clip_scalar(qual, 0.0, 0.49)
                exit_found = True
                break"""

replace = """            if hit_sl and hit_tp:
                # Ambiguous bar - Fallback logic: close proximity to extrema
                # Consider it a win if close price of the ambiguous bar is closer to the high (for longs) / low (for shorts)
                dist_to_high = abs(hh - cc)
                dist_to_low = abs(ll - cc)

                if side == 1:
                    win_condition = dist_to_high < dist_to_low
                else:
                    win_condition = dist_to_low < dist_to_high

                if win_condition:
                    outcomes[i] = OUT_TP
                    returns[i] = activation
                    exit_idxs[i] = j
                    time_elapsed = max(0, tt - entry_t)
                    time_penalty = min(0.15, 0.15 * (time_elapsed / max(limit_ns, 1)))
                    den_sl = max(entry_p * abs(sl), _QUALITY_EPS)
                    mae_ratio = mae_val / den_sl
                    qual = 1.0 - (mae_ratio * 0.5) - time_penalty
                    quality[i] = _clip_scalar(qual, 0.51, 1.0)
                else:
                    outcomes[i] = OUT_SL
                    returns[i] = -sl
                    exit_idxs[i] = j
                    den_tp = max(entry_p * abs(activation), _QUALITY_EPS)
                    qual_raw = (mfe_val / den_tp) * 0.5
                    qual = _soft_squash_pos(qual_raw)
                    quality[i] = _clip_scalar(qual, 0.0, 0.49)

                exit_found = True
                break"""

if search in content:
    content = content.replace(search, replace)
    with open("extreme_price_movements/labeling.py", "w") as f:
        f.write(content)
    print("Successfully replaced ambiguous logic in labeling.py")
else:
    print("Could not find search block in labeling.py")
