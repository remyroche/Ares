with open("extreme_price_movements/offline_optimisers/params_store.py", "r") as f:
    content = f.read()

import re
content = re.sub(
    r"import pandas as pd\s*\n\s*df = pd.read_csv\(path\)\s*\n\s*if df.empty:\s*\n\s*return \[\]",
    r"import pandas as pd\n    try:\n        df = pd.read_csv(path)\n    except pd.errors.EmptyDataError:\n        return []\n    if df.empty:\n        return []",
    content
)

with open("extreme_price_movements/offline_optimisers/params_store.py", "w") as f:
    f.write(content)
