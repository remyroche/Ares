from typing import Any

from src.config import get_complete_config

from .gateio import GateioExchange
from .mexc import MexcExchange
from .okx import OkxExchange


class ExchangeFactory:
    @staticmethod