with open("extreme_price_movements/training_utils.py", "r") as f:
    content = f.read()

import re

# Add the expansion for base_all and meta_all explicitly
new_audit_returns = """    computed_but_unused = sorted(list(set(all_cols) - global_all))
    configured_but_missing = sorted(list(global_all - set(all_cols)))

    return {
        "computed_but_unused": computed_but_unused,
        "configured_but_missing": configured_but_missing,
        "base_unused": base_unused,
        "meta_unused": meta_unused,
        "stale_orphans": stale_orphans,
        "base_all": sorted(list(base_all)),
        "meta_all": sorted(list(meta_all)),
        "global_all": sorted(list(global_all)),
        "base_long": sorted(list(base_long)),
        "base_short": sorted(list(base_short)),
        "meta_reg": sorted(list(meta_reg)),
        "meta_clf": sorted(list(meta_clf)),
        "meta_mfe": sorted(list(meta_mfe)),
        "meta_mae": sorted(list(meta_mae)),
        "meta_asym": sorted(list(meta_asym))
    }"""

content = re.sub(r'    computed_but_unused = sorted\(list\(set\(all_cols\) - global_all\)\)\n.*', new_audit_returns, content, flags=re.DOTALL)

with open("extreme_price_movements/training_utils.py", "w") as f:
    f.write(content)
