import re

with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

# Make sure skip keys also skips the new features since they're already standardized.
insertion_point = '''        "blowoff_risk_surprise", "exh_qual_surprise",
        "dist_vwap_resid", "dist_ema_fast_resid", "trend_pct_resid",
    }'''

new_skip = '''        "blowoff_risk_surprise", "exh_qual_surprise",
        "dist_vwap_resid", "dist_ema_fast_resid", "trend_pct_resid",
    }

    # Add dynamically generated peer context and TS pct to skip set
    for k in feats.keys():
        if k.startswith("cs_rank_") or k.startswith("ts_pct_"):
            skip_transform_set.add(k)
'''

content = content.replace(insertion_point, new_skip)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
