"""
OKX Exchange Implementation

This module provides a complete OKX exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
"""

import asyncio
import hashlib
import hmac
import time
import base64
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None

from src.interfaces.base_interfaces import MarketData
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

from .base_exchange import BaseExchange


class OkxExchange(BaseExchange):
    """
    OKX exchange implementation following the BaseExchange interface.
    
    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
    - Futures funding rates
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)
        self.logger = system_logger.getChild("OkxExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://www.okx.com"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the OKX exchange client."""
        try:
            if aiohttp is None:
                self.logger.warning("⚠️ aiohttp not available, using mock session")
                self.session = None
                return

            # Initialize aiohttp session with SSL configuration
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Test connection
            await self._test_connection()

            self.logger.info("✅ OKX exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OKX exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to OKX API."""
        try:
            url = f"{self.base_url}/api/v5/public/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", [{}])[0].get("ts")
                    self.logger.info(f"Connected to OKX API (Server time: {server_time})")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate OKX signature."""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        body: str = ""
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to OKX API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            headers.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.password or "",
            })

        try:
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", []) if data.get("code") == "0" else []
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed: {response.status} - {error_text}")
                        return None
            else:
                async with self.session.request(method, url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", []) if data.get("code") == "0" else []
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed: {response.status} - {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw OKX kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # OKX klines format: [timestamp, open, high, low, close, volume, ...]
                if isinstance(item, list):
                    timestamp = datetime.fromtimestamp(int(item[0]) / 1000)
                    open_price = float(item[1])
                    high_price = float(item[2])
                    low_price = float(item[3])
                    close_price = float(item[4])
                    volume = float(item[5])
                else:
                    # Dict format
                    timestamp = self._convert_timestamp(item.get("ts", item.get("timestamp", 0)))
                    open_price = float(item.get("open", 0))
                    high_price = float(item.get("high", 0))
                    low_price = float(item.get("low", 0))
                    close_price = float(item.get("close", 0))
                    volume = float(item.get("vol", item.get("volume", 0)))

                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    interval=interval
                )
                market_data_list.append(market_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert kline data: {e}")
                continue

        return market_data_list

    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol (OKX uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from OKX."""
        params = {
            "instId": symbol.upper(),
            "bar": self._convert_interval(interval),
            "limit": min(limit, 300)  # OKX max limit is 300
        }
        
        data = await self._make_request("GET", "/api/v5/market/candles", params)
        if data:
            # Convert OKX format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": item[6],
                    "trades": item[7],
                    "taker_buy_base": item[8],
                    "taker_buy_quote": item[9]
                })
            return klines
        return []

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from OKX."""
        params = {
            "instId": symbol.upper(),
            "bar": self._convert_interval(interval),
            "before": str(end_time_ms),
            "after": str(start_time_ms),
            "limit": min(limit, 300)
        }
        
        data = await self._make_request("GET", "/api/v5/market/candles", params)
        if data:
            # Convert OKX format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": item[6],
                    "trades": item[7],
                    "taker_buy_base": item[8],
                    "taker_buy_quote": item[9]
                })
            return klines
        return []

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from OKX."""
        params = {
            "instId": symbol.upper(),
            "before": str(end_time_ms),
            "after": str(start_time_ms),
            "limit": min(limit, 100)
        }
        
        data = await self._make_request("GET", "/api/v5/market/history-trades", params)
        if data:
            # Standardize field names
            trades = []
            for item in data:
                trades.append({
                    "timestamp": int(item["ts"]),
                    "price": item["px"],
                    "quantity": item["sz"],
                    "is_buyer_maker": item["side"] == "sell",
                    "trade_id": item["tradeId"]
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from OKX."""
        data = await self._make_request("GET", "/api/v5/account/balance", signed=True)
        if data and len(data) > 0:
            account = data[0]
            return {
                "accountId": account.get("accountId"),
                "totalBalance": account.get("totalEq"),
                "availableBalance": account.get("availEq"),
                "frozenBalance": account.get("frozenBal"),
                "details": account.get("details", [])
            }
        return {}

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on OKX."""
        order_params = {
            "instId": symbol.upper(),
            "tdMode": "cash",  # cash, cross, isolated
            "side": "buy" if side.lower() == "buy" else "sell",
            "ordType": "market" if order_type.upper() == "MARKET" else "limit",
            "sz": str(quantity)
        }
        
        if price is not None and order_type.upper() != "MARKET":
            order_params["px"] = str(price)
            
        if params:
            order_params.update(params)
            
        data = await self._make_request("POST", "/api/v5/trade/order", order_params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/account/positions", params, signed=True)
        
        if data and len(data) > 0:
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("instId", "").upper() == symbol.upper():
                    return {
                        "symbol": position.get("instId"),
                        "size": position.get("pos"),
                        "side": position.get("posSide"),
                        "markPrice": position.get("markPx"),
                        "unrealizedPnl": position.get("upl"),
                        "liquidationPrice": position.get("liqPx"),
                        "margin": position.get("margin"),
                        "notionalUsd": position.get("notionalUsd")
                    }
            return data[0] if data else {}
        
        return {}

    async def get_all_positions(self, inst_type: str = "SPOT") -> list[dict[str, Any]]:
        """
        Get all positions from OKX.
        
        Args:
            inst_type: Instrument type (SPOT, MARGIN, SWAP, FUTURES, OPTION)
        
        Returns:
            List of all positions
        """
        try:
            # Validate instrument type
            valid_types = ["SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION"]
            if inst_type.upper() not in valid_types:
                self.logger.warning(f"Invalid instrument type: {inst_type}. Using SPOT.")
                inst_type = "SPOT"
            
            params = {"instType": inst_type.upper()}
            data = await self._make_request("GET", "/api/v5/account/positions", params, signed=True)
            
            if not data:
                self.logger.warning("No position data received from OKX API")
                return []
            
            positions = []
            for position in data:
                try:
                    # Only include positions with non-zero size
                    position_size = float(position.get("pos", "0"))
                    if position_size != 0:
                        positions.append({
                            "symbol": position.get("instId"),
                            "instType": position.get("instType"),
                            "size": str(position_size),
                            "side": position.get("posSide"),
                            "markPrice": position.get("markPx"),
                            "avgPrice": position.get("avgPx"),
                            "unrealizedPnl": position.get("upl"),
                            "unrealizedPnlRatio": position.get("uplRatio"),
                            "liquidationPrice": position.get("liqPx"),
                            "margin": position.get("margin"),
                            "notionalUsd": position.get("notionalUsd"),
                            "marginRatio": position.get("mgnRatio"),
                            "marginMode": position.get("mgnMode"),
                            "interest": position.get("interest"),
                            "lastUpdateTime": position.get("uTime"),
                            "openTime": position.get("cTime"),
                            "leverage": position.get("lever"),
                            "delta": position.get("deltaBS"),
                            "gamma": position.get("gammaBS"),
                            "theta": position.get("thetaBS"),
                            "vega": position.get("vegaBS")
                        })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error parsing position data: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(positions)} positions for {inst_type}")
            return positions
            
        except Exception as e:
            self.logger.error(f"Error fetching all positions: {e}")
            return []

    async def get_position_by_symbol(self, symbol: str, inst_type: str = "SPOT") -> dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            inst_type: Instrument type (SPOT, MARGIN, SWAP, FUTURES, OPTION)
        
        Returns:
            Position information for the symbol
        """
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                self.logger.error("Invalid symbol provided")
                return {}
            
            valid_types = ["SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION"]
            if inst_type.upper() not in valid_types:
                self.logger.warning(f"Invalid instrument type: {inst_type}. Using SPOT.")
                inst_type = "SPOT"
            
            params = {
                "instId": symbol.upper(),
                "instType": inst_type.upper()
            }
            data = await self._make_request("GET", "/api/v5/account/positions", params, signed=True)
            
            if not data or len(data) == 0:
                self.logger.info(f"No position found for {symbol}")
                return {}
            
            position = data[0]
            return {
                "symbol": position.get("instId"),
                "instType": position.get("instType"),
                "size": position.get("pos"),
                "side": position.get("posSide"),
                "markPrice": position.get("markPx"),
                "avgPrice": position.get("avgPx"),
                "unrealizedPnl": position.get("upl"),
                "unrealizedPnlRatio": position.get("uplRatio"),
                "liquidationPrice": position.get("liqPx"),
                "margin": position.get("margin"),
                "notionalUsd": position.get("notionalUsd"),
                "marginRatio": position.get("mgnRatio"),
                "marginMode": position.get("mgnMode"),
                "interest": position.get("interest"),
                "lastUpdateTime": position.get("uTime"),
                "openTime": position.get("cTime"),
                "leverage": position.get("lever"),
                "delta": position.get("deltaBS"),
                "gamma": position.get("gammaBS"),
                "theta": position.get("thetaBS"),
                "vega": position.get("vegaBS")
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return {}

    async def get_position_history(
        self, 
        symbol: str | None = None, 
        inst_type: str = "SPOT",
        after: str | None = None,
        before: str | None = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get position history from OKX.
        
        Args:
            symbol: Trading symbol (optional)
            inst_type: Instrument type (SPOT, MARGIN, SWAP, FUTURES, OPTION)
            after: Pagination of data to return records earlier than the requested ts
            before: Pagination of data to return records newer than the requested ts
            limit: Number of results per request (max 100)
        
        Returns:
            List of historical position changes
        """
        try:
            # Validate inputs
            valid_types = ["SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION"]
            if inst_type.upper() not in valid_types:
                self.logger.warning(f"Invalid instrument type: {inst_type}. Using SPOT.")
                inst_type = "SPOT"
            
            if limit <= 0 or limit > 100:
                self.logger.warning(f"Invalid limit: {limit}. Using 100.")
                limit = 100
            
            params = {
                "instType": inst_type.upper(),
                "limit": limit
            }
            
            if symbol:
                params["instId"] = symbol.upper()
            if after:
                params["after"] = str(after)
            if before:
                params["before"] = str(before)
                
            data = await self._make_request("GET", "/api/v5/account/positions-history", params, signed=True)
            
            if not data:
                self.logger.info("No position history data received")
                return []
            
            history = []
            for record in data:
                try:
                    history.append({
                        "symbol": record.get("instId"),
                        "instType": record.get("instType"),
                        "size": record.get("pos"),
                        "side": record.get("posSide"),
                        "markPrice": record.get("markPx"),
                        "avgPrice": record.get("avgPx"),
                        "unrealizedPnl": record.get("upl"),
                        "unrealizedPnlRatio": record.get("uplRatio"),
                        "liquidationPrice": record.get("liqPx"),
                        "margin": record.get("margin"),
                        "notionalUsd": record.get("notionalUsd"),
                        "marginRatio": record.get("mgnRatio"),
                        "marginMode": record.get("mgnMode"),
                        "interest": record.get("interest"),
                        "lastUpdateTime": record.get("uTime"),
                        "openTime": record.get("cTime"),
                        "leverage": record.get("lever"),
                        "delta": record.get("deltaBS"),
                        "gamma": record.get("gammaBS"),
                        "theta": record.get("thetaBS"),
                        "vega": record.get("vegaBS"),
                        "changeTime": record.get("cTime"),
                        "changeType": record.get("type")
                    })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error parsing history record: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(history)} position history records")
            return history
            
        except Exception as e:
            self.logger.error(f"Error fetching position history: {e}")
            return []

    async def get_position_margin(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get position margin information from OKX.
        
        Args:
            symbol: Trading symbol (optional)
        
        Returns:
            Position margin information
        """
        try:
            params = {}
            if symbol:
                if not isinstance(symbol, str):
                    self.logger.error("Invalid symbol type")
                    return {}
                params["instId"] = symbol.upper()
                
            data = await self._make_request("GET", "/api/v5/account/positions-margin", params, signed=True)
            
            if not data or len(data) == 0:
                self.logger.info("No margin data received")
                return {}
            
            margin_info = data[0]
            return {
                "symbol": margin_info.get("instId"),
                "instType": margin_info.get("instType"),
                "margin": margin_info.get("margin"),
                "marginRatio": margin_info.get("mgnRatio"),
                "marginMode": margin_info.get("mgnMode"),
                "leverage": margin_info.get("lever"),
                "notionalUsd": margin_info.get("notionalUsd"),
                "unrealizedPnl": margin_info.get("upl"),
                "unrealizedPnlRatio": margin_info.get("uplRatio"),
                "liquidationPrice": margin_info.get("liqPx"),
                "lastUpdateTime": margin_info.get("uTime")
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching position margin: {e}")
            return {}

    async def get_position_funding(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Get position funding information from OKX.
        
        Args:
            symbol: Trading symbol (optional)
        
        Returns:
            List of funding information
        """
        try:
            params = {}
            if symbol:
                if not isinstance(symbol, str):
                    self.logger.error("Invalid symbol type")
                    return []
                params["instId"] = symbol.upper()
                
            data = await self._make_request("GET", "/api/v5/account/bills-details", params, signed=True)
            
            if not data:
                self.logger.info("No funding data received")
                return []
            
            funding_info = []
            for record in data:
                try:
                    if record.get("type") == "funding_fee":
                        funding_info.append({
                            "symbol": record.get("instId"),
                            "type": record.get("type"),
                            "amount": record.get("bal"),
                            "balance": record.get("balChg"),
                            "currency": record.get("ccy"),
                            "fee": record.get("fee"),
                            "timestamp": record.get("ts"),
                            "orderId": record.get("ordId"),
                            "clientOrderId": record.get("clOrdId")
                        })
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error parsing funding record: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(funding_info)} funding records")
            return funding_info
            
        except Exception as e:
            self.logger.error(f"Error fetching position funding: {e}")
            return []

    async def get_position_risk_metrics(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get comprehensive position risk metrics.
        
        Args:
            symbol: Trading symbol (optional)
        
        Returns:
            Risk metrics for positions
        """
        try:
            positions = await self.get_all_positions() if not symbol else [await self.get_position_by_symbol(symbol)]
            
            if not positions:
                self.logger.info("No positions found for risk calculation")
                return {}
            
            total_unrealized_pnl = 0
            total_notional = 0
            total_margin = 0
            max_leverage = 0
            risk_positions = []
            
            for position in positions:
                try:
                    unrealized_pnl = float(position.get("unrealizedPnl", 0))
                    notional = float(position.get("notionalUsd", 0))
                    margin = float(position.get("margin", 0))
                    leverage = float(position.get("leverage", 0))
                    
                    total_unrealized_pnl += unrealized_pnl
                    total_notional += notional
                    total_margin += margin
                    max_leverage = max(max_leverage, leverage)
                    
                    # Calculate position-specific risk metrics
                    position_risk = {
                        "symbol": position.get("symbol"),
                        "size": position.get("size"),
                        "side": position.get("side"),
                        "unrealizedPnl": unrealized_pnl,
                        "notionalUsd": notional,
                        "margin": margin,
                        "leverage": leverage,
                        "liquidationPrice": position.get("liquidationPrice"),
                        "marginRatio": position.get("marginRatio"),
                        "risk_score": abs(unrealized_pnl) / max(notional, 1) if notional > 0 else 0
                    }
                    risk_positions.append(position_risk)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error calculating risk for position {position.get('symbol', 'unknown')}: {e}")
                    continue
            
            # Sort by risk score (highest first)
            risk_positions.sort(key=lambda x: x["risk_score"], reverse=True)
            
            return {
                "totalPositions": len(positions),
                "totalUnrealizedPnl": total_unrealized_pnl,
                "totalNotionalUsd": total_notional,
                "totalMargin": total_margin,
                "maxLeverage": max_leverage,
                "averageLeverage": sum(p.get("leverage", 0) for p in positions) / len(positions) if positions else 0,
                "portfolioRiskScore": abs(total_unrealized_pnl) / max(total_notional, 1) if total_notional > 0 else 0,
                "positions": risk_positions,
                "highRiskPositions": [p for p in risk_positions if p["risk_score"] > 0.1],
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk metrics: {e}")
            return {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/trade/orders-pending", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        data = await self._make_request("POST", "/api/v5/trade/cancel-order", params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        data = await self._make_request("GET", "/api/v5/trade/order", params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to OKX format."""
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6Hutc",
            "12h": "12Hutc",
            "1d": "1Dutc",
            "3d": "3Dutc",
            "1w": "1Wutc",
            "1M": "1Mutc"
        }
        return interval_map.get(interval, "1m")

    def _get_interval_ms(self, interval: str) -> int:
        """Get interval duration in milliseconds."""
        interval_map = {
            "1m": 60000,
            "3m": 180000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "2h": 7200000,
            "4h": 14400000,
            "6h": 21600000,
            "12h": 43200000,
            "1d": 86400000,
            "3d": 259200000,
            "1w": 604800000,
            "1M": 2592000000
        }
        return interval_map.get(interval, 60000)

    # Additional methods for live trading
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/market/ticker", params)
        if data and len(data) > 0:
            ticker = data[0]
            return {
                "symbol": ticker.get("instId"),
                "last": ticker.get("last"),
                "bid": ticker.get("bidPx"),
                "ask": ticker.get("askPx"),
                "high": ticker.get("high24h"),
                "low": ticker.get("low24h"),
                "volume": ticker.get("vol24h"),
                "quoteVolume": ticker.get("volCcy24h"),
                "change": ticker.get("chg"),
                "changePercent": ticker.get("chgRate"),
                "timestamp": ticker.get("ts")
            }
        return {}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades."""
        params = {
            "instId": symbol.upper(),
            "limit": min(limit, 100)
        }
        data = await self._make_request("GET", "/api/v5/market/trades", params)
        if data:
            trades = []
            for item in data:
                trades.append({
                    "timestamp": int(item["ts"]),
                    "price": item["px"],
                    "quantity": item["sz"],
                    "side": item["side"],
                    "trade_id": item["tradeId"]
                })
            return trades
        return []
    
    async def get_funding_rate(self, symbol: str | None = None) -> dict[str, Any]:
        """Get funding rate information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/funding-rate", params)
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_open_interest(self, symbol: str | None = None) -> dict[str, Any]:
        """Get open interest information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/open-interest", params)
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        data = await self._make_request("GET", "/api/v5/public/time")
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_instruments(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get instruments information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/instruments", params)
        return data if isinstance(data, list) else []
    
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        instruments = await self.get_instruments(symbol)
        if instruments and len(instruments) > 0:
            return instruments[0]
        return {}
    
    async def get_klines_stream(self, symbol: str, interval: str, callback) -> None:
        """Stream klines data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                klines = await self.get_klines(symbol, interval, limit=1)
                if klines:
                    await callback(klines[0])
                await asyncio.sleep(60)  # Poll every minute
            except Exception as e:
                self.logger.error(f"Error in klines stream: {e}")
                await asyncio.sleep(60)
    
    async def get_ticker_stream(self, symbol: str, callback) -> None:
        """Stream ticker data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                ticker = await self.get_ticker(symbol)
                if ticker:
                    await callback(ticker)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in ticker stream: {e}")
                await asyncio.sleep(1)
    
    async def get_trade_stream(self, symbol: str, callback) -> None:
        """Stream trade data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                trades = await self.get_recent_trades(symbol, limit=10)
                for trade in trades:
                    await callback(trade)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in trade stream: {e}")
                await asyncio.sleep(1)
    
    async def get_orderbook_stream(self, symbol: str, callback) -> None:
        """Stream orderbook data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                orderbook = await self.get_order_book(symbol, limit=20)
                if orderbook:
                    await callback(orderbook)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in orderbook stream: {e}")
                await asyncio.sleep(1)

    async def get_positions_stream(self, callback, inst_type: str = "SPOT") -> None:
        """Stream position updates (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                positions = await self.get_all_positions(inst_type)
                if positions:
                    await callback(positions)
                await asyncio.sleep(5)  # Poll every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in positions stream: {e}")
                await asyncio.sleep(5)

    async def get_position_risk_stream(self, symbol: str, callback) -> None:
        """Stream position risk updates for a specific symbol."""
        while True:
            try:
                risk = await self.get_position_risk_metrics(symbol)
                if risk:
                    await callback(risk)
                await asyncio.sleep(2)  # Poll every 2 seconds
            except Exception as e:
                self.logger.error(f"Error in position risk stream: {e}")
                await asyncio.sleep(2)

    async def get_position_summary(self, inst_type: str = "SPOT") -> dict[str, Any]:
        """
        Get a comprehensive position summary.
        
        Args:
            inst_type: Instrument type (SPOT, MARGIN, SWAP, FUTURES, OPTION)
        
        Returns:
            Position summary with key metrics
        """
        positions = await self.get_all_positions(inst_type)
        
        if not positions:
            return {
                "totalPositions": 0,
                "totalValue": 0,
                "totalUnrealizedPnl": 0,
                "totalMargin": 0,
                "averageLeverage": 0,
                "maxLeverage": 0,
                "longPositions": 0,
                "shortPositions": 0,
                "positions": []
            }
        
        total_value = 0
        total_unrealized_pnl = 0
        total_margin = 0
        long_positions = 0
        short_positions = 0
        leverages = []
        
        for position in positions:
            notional = float(position.get("notionalUsd", 0))
            unrealized_pnl = float(position.get("unrealizedPnl", 0))
            margin = float(position.get("margin", 0))
            leverage = float(position.get("leverage", 0))
            side = position.get("side", "")
            
            total_value += notional
            total_unrealized_pnl += unrealized_pnl
            total_margin += margin
            leverages.append(leverage)
            
            if side.lower() in ["long", "net"]:
                long_positions += 1
            elif side.lower() == "short":
                short_positions += 1
        
        return {
            "totalPositions": len(positions),
            "totalValue": total_value,
            "totalUnrealizedPnl": total_unrealized_pnl,
            "totalMargin": total_margin,
            "averageLeverage": sum(leverages) / len(leverages) if leverages else 0,
            "maxLeverage": max(leverages) if leverages else 0,
            "longPositions": long_positions,
            "shortPositions": short_positions,
            "netExposure": total_value * (1 if total_unrealized_pnl >= 0 else -1),
            "marginUtilization": (total_margin / max(total_value, 1)) * 100 if total_value > 0 else 0,
            "positions": positions,
            "timestamp": int(time.time() * 1000)
        }

    async def get_position_alerts(self, risk_threshold: float = 0.1) -> list[dict[str, Any]]:
        """
        Get position alerts based on risk thresholds.
        
        Args:
            risk_threshold: Risk threshold for alerts (0.1 = 10%)
        
        Returns:
            List of position alerts
        """
        risk_metrics = await self.get_position_risk_metrics()
        alerts = []
        
        if not risk_metrics:
            return alerts
        
        # Check for high-risk positions
        high_risk_positions = risk_metrics.get("highRiskPositions", [])
        for position in high_risk_positions:
            alerts.append({
                "type": "HIGH_RISK",
                "symbol": position["symbol"],
                "risk_score": position["risk_score"],
                "unrealized_pnl": position["unrealizedPnl"],
                "notional_usd": position["notionalUsd"],
                "leverage": position["leverage"],
                "message": f"High risk position: {position['symbol']} with risk score {position['risk_score']:.2%}"
            })
        
        # Check for high leverage positions
        for position in risk_metrics.get("positions", []):
            leverage = position.get("leverage", 0)
            if leverage > 10:  # Alert for leverage > 10x
                alerts.append({
                    "type": "HIGH_LEVERAGE",
                    "symbol": position["symbol"],
                    "leverage": leverage,
                    "message": f"High leverage position: {position['symbol']} with {leverage}x leverage"
                })
        
        # Check for portfolio-level risks
        portfolio_risk = risk_metrics.get("portfolioRiskScore", 0)
        if portfolio_risk > risk_threshold:
            alerts.append({
                "type": "PORTFOLIO_RISK",
                "portfolio_risk_score": portfolio_risk,
                "message": f"Portfolio risk exceeds threshold: {portfolio_risk:.2%}"
            })
        
        return alerts

    async def calculate_position_size(
        self, 
        symbol: str, 
        risk_amount: float, 
        entry_price: float, 
        stop_loss_price: float,
        leverage: float = 1.0
    ) -> dict[str, Any]:
        """
        Calculate optimal position size based on risk management.
        
        Args:
            symbol: Trading symbol
            risk_amount: Amount willing to risk
            entry_price: Entry price
            stop_loss_price: Stop loss price
            leverage: Leverage to use
        
        Returns:
            Position size calculation
        """
        price_diff = abs(entry_price - stop_loss_price)
        risk_per_unit = price_diff / entry_price
        
        if risk_per_unit == 0:
            return {"error": "Invalid price difference"}
        
        # Calculate position size based on risk
        position_value = risk_amount / risk_per_unit
        position_size = position_value / entry_price
        
        # Apply leverage
        leveraged_size = position_size * leverage
        leveraged_value = position_value * leverage
        
        return {
            "symbol": symbol,
            "risk_amount": risk_amount,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "risk_per_unit": risk_per_unit,
            "position_size": position_size,
            "position_value": position_value,
            "leveraged_size": leveraged_size,
            "leveraged_value": leveraged_value,
            "leverage": leverage,
            "risk_reward_ratio": price_diff / (entry_price * 0.02) if entry_price > 0 else 0,  # Assuming 2% target
            "margin_required": leveraged_value / leverage if leverage > 0 else leveraged_value
        }

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("OKX exchange connection closed")


# Factory function for creating OKX exchange instances
def create_okx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> OkxExchange:
    """Create a new OKX exchange instance."""
    return OkxExchange(api_key, api_secret, trade_symbol, password)
