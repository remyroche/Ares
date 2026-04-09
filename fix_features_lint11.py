with open("extreme_price_movements/features.py", "r") as f:
    content = f.read()

content = content.replace("from numba import njit", "")
content = content.replace("import numpy as np", "")
content = content.replace("import pandas as pd", "import pandas as pd\nimport numpy as np\nfrom numba import njit\nfrom numba import prange\nfrom typing import Dict, List, Tuple\n")
content = content.replace("from typing import Dict, List, Tuple", "")
content = content.replace("from numba import prange", "")

with open("extreme_price_movements/features.py", "w") as f:
    f.write(content)
