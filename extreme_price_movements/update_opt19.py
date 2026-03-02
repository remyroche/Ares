import re

with open("extreme_price_movements/labeling.py", "r") as f:
    content = f.read()

search_missing_import = "import pandas as pd"
replace_missing_import = """import pandas as pd
import warnings
import ccxt
from extreme_price_movements.hf_data_loader import get_15m_ohlcv"""

content = content.replace(search_missing_import, replace_missing_import)

with open("extreme_price_movements/labeling.py", "w") as f:
    f.write(content)
