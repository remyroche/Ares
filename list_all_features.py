import re

def get_config_keys():
    with open('extreme_price_movements/config.py', 'r') as f:
        config_content = f.read()

    list_names = [
        "FEATURE_KEYS_15M_OHLCV", "neutral_feature_keys", "MODEL_FEATURES",
        "RIDGE_FEATURE_COLS", "CONTINUOUS_TRIGGER_COLS", "CONTINUOUS_LOCATION_COLS",
        "HELPER_BASE_FEATURES", "TEST_FEATURE_KEYS",
        "position_sizer_feature_priority", "limit_offset_sizer",
        "position_sizer_regime_feature_keys", "exh_feature_keys",
        "spike_feature_keys", "tf_feature_keys", "mr_feature_keys",
        "meta_feature_keys", "mr_meta_feature_keys", "tf_meta_feature_keys",
        "causal_cols"
    ]

    keys = []
    for list_name in list_names:
        pattern = list_name + r'\s*=\s*\[(.*?)\]'
        match = re.search(pattern, config_content, re.DOTALL)
        if match:
            elements = re.findall(r'"([^"]+)"', match.group(1))
            keys.extend(elements)

    return set(keys)

def get_feature_keys():
    with open('extreme_price_movements/features.py', 'r') as f:
        content = f.read()

    static_keys = set(re.findall(r'feats\["([^"]+)"\]', content))

    # 2. Add keys created from f-strings manually to simulate evaluation
    # These represent the loops in features.py
    f_keys = set()

    for H in [2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 28, 48, 72, 120]:
        f_keys.add(f"ret{H}h")

    for d in [0.4, 0.5, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        for h in [1, 2, 4, 8]:
            f_keys.add(f"ffd_diff_{h}_{d_tag}")
        f_keys.add(f"ffd_ema_spread_{d_tag}")
        f_keys.add(f"ffd_rv_12_{d_tag}")
        f_keys.add(f"ffd_rv_24_{d_tag}")
        f_keys.add(f"ffd_z_24_{d_tag}")
        f_keys.add(f"ffd_range_24_{d_tag}")

    for d in [0.4, 0.5]:
        d_tag = f"{int(round(d * 10)):02d}"
        for w in [12, 24]:
            f_keys.add(f"ffd_slope_{d_tag}_{w}")
        f_keys.add(f"ffd_mr_z_{d_tag}")

    for d in [0.5, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        f_keys.add(f"ffd_d1_{d_tag}")
        f_keys.add(f"ffd_d4_{d_tag}")

    for d in [0.4]:
        d_tag = f"{int(round(d * 10)):02d}"
        for w in [12, 24]:
            f_keys.add(f"ffd_ctx_slope_{d_tag}_{w}")

    for H in [2, 3, 4, 5, 6]:
        f_keys.add(f"ret{H}h")
    for H in [8, 10, 12, 16, 20, 24, 28, 48, 72, 120]:
        f_keys.add(f"ret{H}h")

    for d in [0.4, 0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        f_keys.add(f"ffd_rv_2h_{d_tag}")
        f_keys.add(f"ffd_rv_6h_{d_tag}")
        f_keys.add(f"ffd_rv_24h_{d_tag}")

        f_keys.add(f"ffd_vol_price_corr_10h_{d_tag}")

        for k in [12, 24, 48]:
            f_keys.add(f"ffd_donch_dist_{d_tag}_{k}")

        f_keys.add(f"ffd_amihud_{d_tag}")
        f_keys.add(f"ffd_vol_range_shock_{d_tag}")

    for d in [0.6]:
        d_tag = f"{int(round(d * 10)):02d}"
        f_keys.add(f"ffd_accel_{d_tag}")
        f_keys.add(f"ffd_z_{d_tag}")
        f_keys.add(f"ffd_atr_expansion_{d_tag}")
        f_keys.add(f"ffd_cvar_5pct_{d_tag}")

    for d in [0.4]:
        d_tag = f"{int(round(d * 10)):02d}"
        f_keys.add(f"ffd_dist_ema_fast_{d_tag}")
        f_keys.add(f"ffd_dist_ema_slow_{d_tag}")

    for k in [2, 4, 6, 8, 12, 24, 48, 72, 120]:
        f_keys.add(f"donch_dist_{k}")
        f_keys.add(f"pullback_{k}")
        if k >= 48:
            f_keys.add(f"dist_from_high_{k}h")
            f_keys.add(f"dist_from_low_{k}h")

    for k_trend in [48, 72, 120]:
        f_keys.add(f"trend_slope_{k_trend}h")
        f_keys.add(f"trend_accel_{k_trend}h")

    for k in [2, 4, 8]:
        f_keys.add(f"ft_{k}")
        f_keys.add(f"failure_{k}")

    for k in [3, 6]:
        f_keys.add(f"evr_{k}")

    for n in [10, 16, 24]:
        f_keys.add(f"ker_{n}")

    for n in [14, 21, 34]:
        f_keys.add(f"vortex_diff_{n}")

    for n in [7, 10, 14]:
        f_keys.add(f"adx_{n}")
        f_keys.add(f"adx_di_plus_{n}")
        f_keys.add(f"adx_di_minus_{n}")
        f_keys.add(f"adx_{n}_gt25")
        f_keys.add(f"adx_{n}_slope")

    for n in [12, 24, 96]:
        f_keys.add(f"dist_vwap_{n}_atr")
        f_keys.add(f"trapped_longs_{n}")

    for n in [12, 24]:
        f_keys.add(f"range_norm_{n}")
        f_keys.add(f"sv_imb_{n}")
        f_keys.add(f"press_{n}")
        f_keys.add(f"impact_{n}")
        f_keys.add(f"ts_{n}")
        f_keys.add(f"prog_eff_{n}")
        f_keys.add(f"pers_{n}")
        f_keys.add(f"hh_count_{n}")
        f_keys.add(f"ll_count_{n}")
        f_keys.add(f"skew_{n}")
        f_keys.add(f"climax_range_{n}")
        f_keys.add(f"climax_vol_{n}")
        f_keys.add(f"z_vwap_{n}")
        f_keys.add(f"z_r_{n}")
        f_keys.add(f"bb_pos_{n}")

    # Also grab keys added via `_liq_feats_temp` and `gate_interactions` and others
    temp_keys = re.findall(r'_liq_feats_temp\["([^"]+)"\]', content)
    static_keys.update(temp_keys)

    return static_keys.union(f_keys)

def get_intraday_library_keys():
    with open('extreme_price_movements/intraday_crypto_library.py', 'r') as f:
        content = f.read()
    keys = set(re.findall(r'out\["([^"]+)"\]', content))
    return keys

config_keys = get_config_keys()
feat_keys = get_feature_keys()
lib_keys = get_intraday_library_keys()

all_generated_keys = feat_keys.union(lib_keys)

missing_in_features = config_keys - all_generated_keys

# Let's inspect the `gate_interactions` prefixes which create keys dynamically
gate_prefixes = ["accept_dir2h", "reject_dir2h", "tfq_dir2h", "mrq_dir2h"]
# `add_gate_interaction_panel` presumably adds `_prod`, `_abs_prod`, `_signed_mag`
gate_suffixes = ["_prod", "_abs_prod", "_signed_mag"]
for prefix in gate_prefixes:
    for suffix in gate_suffixes:
        all_generated_keys.add(prefix + suffix)

missing_in_features = config_keys - all_generated_keys

missing_filtered = [k for k in missing_in_features if not k.startswith("LOC_") and not k.startswith("LONG_") and not k.startswith("SHORT_") and not k.startswith("vp_")]

print("\nKeys in config but apparently not created in features.py:")
for k in sorted(missing_filtered):
    print(f"  - {k}")
