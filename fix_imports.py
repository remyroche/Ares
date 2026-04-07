with open("extreme_price_movements/soft_labels.py", "r") as f:
    content = f.read()

content = "import numpy as np\nfrom extreme_price_movements.utils import tprint\n" + content
with open("extreme_price_movements/soft_labels.py", "w") as f:
    f.write(content)
