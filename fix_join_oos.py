import pandas as pd

diff_text = """
<<<<<<< SEARCH
    for col in ("oos_expected_net_bps", "oos_expected_net_for_same_policy", "oos_selected"):
        asof_col = col + "_asof" if col in exact_keys else col
        if asof_col in merged.columns:
            mask = merged[asof_col].notna()
            if mask.any():
                out.loc[merged.index[mask], col] = merged.loc[mask, asof_col]
=======
    for col in ("oos_expected_net_bps", "oos_expected_net_for_same_policy", "oos_selected"):
        asof_col = col + "_asof" if col in out.columns and col in oos.columns else col
        if asof_col in merged.columns:
            mask = merged[asof_col].notna()
            if mask.any():
                out.loc[merged.index[mask], col] = merged.loc[mask, asof_col]
>>>>>>> REPLACE
"""

import urllib.request
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "http://localhost:8000/replace_with_git_merge_diff"
data = {
    "filepath": "extreme_price_movements/inference/live_replay.py",
    "merge_diff": diff_text
}
req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as response:
    print(response.read().decode())
