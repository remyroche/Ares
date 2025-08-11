# src/types/base_types.py

"""
Base type definitions for the Ares trading system.
Provides fundamental types used throughout the system.
"""

from datetime import datetime
from decimal import Decimal
from typing import NewType

# Fundamental trading types
Timestamp = NewType("Timestamp", datetime)
Symbol = NewType("Symbol", str)
Price = NewType("Price", float | Decimal)
Volume = NewType("Volume", float | Decimal)
Percentage = NewType("Percentage", float)  # 0.0 to 1.0
Score = NewType("Score", float)  # Typically 0.0 to 1.0
Interval = NewType("Interval", str)  # e.g., "1m", "5m", "1h"

# Identifiers
OrderId = NewType("OrderId", str)
TradeId = NewType("TradeId", str)
PositionId = NewType("PositionId", str)
ModelId = NewType("ModelId", str)
UserId = NewType("UserId", str)
SessionId = NewType("SessionId", str)

# Numeric constraints
LeverageMultiplier = NewType("LeverageMultiplier", float)  # 1.0 to 100.0
RiskScore = NewType("RiskScore", float)  # 0.0 to 1.0 where 1.0 is max risk
ConfidenceLevel = NewType("ConfidenceLevel", float)  # 0.0 to 1.0
