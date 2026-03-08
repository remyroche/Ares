"""
Duplicated tprint functions from src/utils/tprint.py
Required for extreme_price_movements/ to be self-contained.
"""

import sys
import time
from datetime import datetime, timezone

def tprint(msg):
    """
    Simple timestamped print function - duplicated from src/utils/tprint.py
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {msg}")

def tprint_warning(msg):
    """
    Simple warning print function - duplicated from src/utils/tprint.py
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] WARNING: {msg}", file=sys.stderr)
