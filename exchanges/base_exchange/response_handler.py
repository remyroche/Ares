"""
Exchange Response Handler

Handles responses from exchanges and routes them back to the appropriate recipients.
Provides aggregation, filtering, and callback mechanisms.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class ResponseType(Enum):
    """Types of responses from exchanges"""
    ORDER_EXECUTION = "order_execution"
    ORDER_STATUS = "order_status"
    MARKET_DATA = "market_data"
    ACCOUNT_INFO = "account_info"
    POSITION_INFO = "position_info"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SYSTEM_STATUS = "system_status"


class ResponseStatus(Enum):
    """Status of response processing"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY_NEEDED = "retry_needed"


@dataclass
class ExchangeResponse:
    """Base response structure from exchanges"""
    response_id: str
    exchange_name: str
    response_type: ResponseType
    status: ResponseStatus
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    processing_time: float = 0.0  # Time taken to process response


@dataclass
class OrderExecutionResponse(ExchangeResponse):
    """Response from order execution"""
    order_id: str = ""
    exchange_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: float = 0.0
    price: Optional[float] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: str = ""


@dataclass
class MarketDataResponse(ExchangeResponse):
    """Response containing market data"""
    symbol: str = ""
    data_type: str = ""  # ticker, kline, trade, orderbook, etc.
    data: Dict[str, Any] = field(default_factory=dict)
    interval: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class AggregatedResponse:
    """Aggregated response from multiple exchanges"""
    response_id: str
    request_id: str
    response_type: ResponseType
    aggregated_data: Dict[str, Any] = field(default_factory=dict)
    individual_responses: Dict[str, ExchangeResponse] = field(default_factory=dict)
    successful_exchanges: List[str] = field(default_factory=list)
    failed_exchanges: List[str] = field(default_factory=list)
    aggregation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ExchangeResponseHandler:
    """
    Handles responses from exchanges and routes them back to the appropriate recipients.
    Provides aggregation, filtering, and callback mechanisms.
    """

    def __init__(self):
        self.logger = logging.getLogger("ExchangeResponseHandler")
        self.response_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.pending_responses: Dict[str, Dict[str, ExchangeResponse]] = defaultdict(dict)
        self.aggregated_responses: Dict[str, AggregatedResponse] = {}
        self._response_processing_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Start the response handler"""
        if self._running:
            return

        self._running = True
        self.logger.info("Exchange response handler started")

    async def stop(self) -> None:
        """Stop the response handler"""
        if not self._running:
            return

        self._running = False

        # Cancel any pending tasks
        for task in self._response_processing_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._response_processing_tasks:
            await asyncio.gather(*self._response_processing_tasks.values(), return_exceptions=True)

        self._response_processing_tasks.clear()
        self.logger.info("Exchange response handler stopped")

    def register_response_callback(
        self,
        response_type: ResponseType,
        callback: Callable[[ExchangeResponse], Awaitable[None]]
    ) -> str:
        """
        Register a callback for a specific response type.

        Args:
            response_type: Type of response to handle
            callback: Async callback function

        Returns:
            Callback ID for unregistration
        """
        callback_id = f"{response_type.value}_{len(self.response_callbacks[response_type])}"
        self.response_callbacks[response_type].append(callback)
        self.logger.debug(f"Registered callback {callback_id} for {response_type.value}")
        return callback_id

    def unregister_response_callback(
        self,
        response_type: ResponseType,
        callback_id: str
    ) -> bool:
        """
        Unregister a response callback.

        Args:
            response_type: Type of response
            callback_id: ID of the callback to remove

        Returns:
            True if callback was removed, False otherwise
        """
        if response_type in self.response_callbacks:
            # Find and remove the callback
            callbacks = self.response_callbacks[response_type]
            for i, callback in enumerate(callbacks):
                if hasattr(callback, '__name__'):
                    # Try to match by name (simplified)
                    continue

            # For now, just return True (in real implementation, proper ID matching)
            return True

        return False

    async def handle_response(
        self,
        response: ExchangeResponse,
        aggregate_responses: bool = True,
        aggregate_timeout: float = 5.0
    ) -> None:
        """
        Handle a response from an exchange.

        Args:
            response: Response from the exchange
            aggregate_responses: Whether to aggregate responses
            aggregate_timeout: Timeout for aggregation in seconds
        """
        try:
            # Store the response
            request_id = response.correlation_id or response.response_id
            self.pending_responses[request_id][response.exchange_name] = response

            # Call type-specific callbacks
            await self._call_response_callbacks(response)

            # Aggregate if requested
            if aggregate_responses:
                await self._aggregate_responses(request_id, aggregate_timeout)

        except Exception as e:
            self.logger.error(f"Error handling response: {e}")

    async def handle_order_response(
        self,
        exchange_name: str,
        order_id: str,
        response_data: Dict[str, Any]
    ) -> None:
        """
        Handle an order response from a specific exchange.

        Args:
            exchange_name: Name of the exchange
            order_id: Order ID
            response_data: Response data from the exchange
        """
        try:
            # Create order execution response
            order_response = OrderExecutionResponse(
                response_id=f"order_{order_id}_{exchange_name}_{int(datetime.now().timestamp() * 1000)}",
                exchange_name=exchange_name,
                response_type=ResponseType.ORDER_EXECUTION,
                status=self._map_response_status(response_data),
                data=response_data,
                order_id=order_id,
                exchange_order_id=response_data.get("orderId"),
                symbol=response_data.get("symbol", ""),
                side=response_data.get("side", ""),
                order_type=response_data.get("type", ""),
                quantity=float(response_data.get("quantity", 0)),
                price=response_data.get("price"),
                filled_quantity=float(response_data.get("executedQty", 0)),
                remaining_quantity=float(response_data.get("remainingQty", 0)),
                average_fill_price=response_data.get("avgPrice"),
                commission=float(response_data.get("commission", 0)),
                commission_asset=response_data.get("commissionAsset", ""),
                error_message=response_data.get("error")
            )

            await self.handle_response(order_response)

        except Exception as e:
            self.logger.error(f"Error handling order response: {e}")

    async def handle_market_data_response(
        self,
        exchange_name: str,
        data_type: str,
        symbol: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Handle market data response from an exchange.

        Args:
            exchange_name: Name of the exchange
            data_type: Type of market data
            symbol: Trading symbol
            data: Market data
        """
        try:
            market_response = MarketDataResponse(
                response_id=f"market_data_{data_type}_{symbol}_{exchange_name}_{int(datetime.now().timestamp() * 1000)}",
                exchange_name=exchange_name,
                response_type=ResponseType.MARKET_DATA,
                status=ResponseStatus.SUCCESS,
                symbol=symbol,
                data_type=data_type,
                data=data
            )

            await self.handle_response(market_response)

        except Exception as e:
            self.logger.error(f"Error handling market data response: {e}")

    async def _call_response_callbacks(self, response: ExchangeResponse) -> None:
        """Call registered callbacks for a response type"""
        callbacks = self.response_callbacks.get(response.response_type, [])

        for callback in callbacks:
            try:
                await callback(response)
            except Exception as e:
                self.logger.error(f"Error in response callback for {response.response_type.value}: {e}")

    async def _aggregate_responses(
        self,
        request_id: str,
        timeout: float
    ) -> Optional[AggregatedResponse]:
        """
        Aggregate responses for a request.

        Args:
            request_id: ID of the request
            timeout: Timeout for aggregation

        Returns:
            Aggregated response if aggregation is complete
        """
        try:
            # Get pending responses for this request
            responses = self.pending_responses.get(request_id, {})

            if not responses:
                return None

            # Check if we have responses from all expected exchanges
            # In a real implementation, this would check against expected exchanges
            # For now, we'll aggregate after a short delay

            await asyncio.sleep(timeout)

            if request_id not in self.pending_responses:
                return None

            final_responses = self.pending_responses[request_id]

            # Create aggregated response
            successful_exchanges = [
                ex for ex, resp in final_responses.items()
                if resp.status == ResponseStatus.SUCCESS
            ]

            failed_exchanges = [
                ex for ex, resp in final_responses.items()
                if resp.status != ResponseStatus.SUCCESS
            ]

            # Aggregate data based on response type
            aggregated_data = await self._aggregate_data_by_type(final_responses)

            aggregated_response = AggregatedResponse(
                response_id=f"aggregated_{request_id}_{int(datetime.now().timestamp() * 1000)}",
                request_id=request_id,
                response_type=list(final_responses.values())[0].response_type if final_responses else ResponseType.MARKET_DATA,
                aggregated_data=aggregated_data,
                individual_responses=final_responses,
                successful_exchanges=successful_exchanges,
                failed_exchanges=failed_exchanges,
                aggregation_time=timeout
            )

            self.aggregated_responses[request_id] = aggregated_response

            # Clean up pending responses
            del self.pending_responses[request_id]

            self.logger.debug(f"Aggregated responses for request {request_id}")
            return aggregated_response

        except Exception as e:
            self.logger.error(f"Error aggregating responses: {e}")
            return None

    async def _aggregate_data_by_type(self, responses: Dict[str, ExchangeResponse]) -> Dict[str, Any]:
        """
        Aggregate data based on response type.

        Args:
            responses: Responses to aggregate

        Returns:
            Aggregated data
        """
        if not responses:
            return {}

        # Get the response type from the first response
        response_type = list(responses.values())[0].response_type

        if response_type == ResponseType.MARKET_DATA:
            return await self._aggregate_market_data(responses)
        elif response_type == ResponseType.ORDER_EXECUTION:
            return await self._aggregate_order_data(responses)
        elif response_type == ResponseType.ACCOUNT_INFO:
            return await self._aggregate_account_data(responses)
        else:
            return {"responses": list(responses.values())}

    async def _aggregate_market_data(self, responses: Dict[str, ExchangeResponse]) -> Dict[str, Any]:
        """
        Aggregate market data from multiple exchanges.

        Args:
            responses: Market data responses

        Returns:
            Aggregated market data
        """
        all_data = {}
        successful_responses = {}

        for exchange, response in responses.items():
            if response.status == ResponseStatus.SUCCESS:
                market_response = response
                if isinstance(market_response, MarketDataResponse):
                    successful_responses[exchange] = market_response.data
                    all_data.update(market_response.data)

        # Calculate best bid/ask across exchanges
        best_bid = 0
        best_ask = float('inf')
        best_bid_exchange = None
        best_ask_exchange = None

        for exchange, data in successful_responses.items():
            bid = float(data.get("bid", 0))
            ask = float(data.get("ask", 0))

            if bid > best_bid:
                best_bid = bid
                best_bid_exchange = exchange

            if ask < best_ask:
                best_ask = ask
                best_ask_exchange = exchange

        return {
            "aggregated": all_data,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_exchange": best_bid_exchange,
            "best_ask_exchange": best_ask_exchange,
            "spread": best_ask - best_bid if best_ask != float('inf') and best_bid > 0 else 0,
            "exchanges_count": len(successful_responses),
            "successful_exchanges": list(successful_responses.keys())
        }

    async def _aggregate_order_data(self, responses: Dict[str, ExchangeResponse]) -> Dict[str, Any]:
        """
        Aggregate order data from multiple exchanges.

        Args:
            responses: Order responses

        Returns:
            Aggregated order data
        """
        successful_orders = []
        failed_orders = []
        total_filled = 0
        total_quantity = 0

        for exchange, response in responses.items():
            if response.status == ResponseStatus.SUCCESS:
                order_response = response
                if isinstance(order_response, OrderExecutionResponse):
                    successful_orders.append({
                        "exchange": exchange,
                        "order_id": order_response.order_id,
                        "exchange_order_id": order_response.exchange_order_id,
                        "filled_quantity": order_response.filled_quantity,
                        "price": order_response.average_fill_price
                    })
                    total_filled += order_response.filled_quantity
                    total_quantity += order_response.quantity
            else:
                failed_orders.append({
                    "exchange": exchange,
                    "error": response.error_message
                })

        return {
            "successful_orders": successful_orders,
            "failed_orders": failed_orders,
            "total_filled": total_filled,
            "total_quantity": total_quantity,
            "fill_rate": (total_filled / total_quantity * 100) if total_quantity > 0 else 0,
            "successful_exchanges": len(successful_orders),
            "failed_exchanges": len(failed_orders)
        }

    async def _aggregate_account_data(self, responses: Dict[str, ExchangeResponse]) -> Dict[str, Any]:
        """
        Aggregate account data from multiple exchanges.

        Args:
            responses: Account responses

        Returns:
            Aggregated account data
        """
        total_balance = 0
        total_available = 0
        balances_by_exchange = {}

        for exchange, response in responses.items():
            if response.status == ResponseStatus.SUCCESS:
                data = response.data
                balance = float(data.get("totalBalance", 0))
                available = float(data.get("availableBalance", 0))

                total_balance += balance
                total_available += available

                balances_by_exchange[exchange] = {
                    "total_balance": balance,
                    "available_balance": available
                }

        return {
            "total_balance": total_balance,
            "total_available": total_available,
            "balances_by_exchange": balances_by_exchange,
            "exchanges_count": len(balances_by_exchange)
        }

    def _map_response_status(self, response_data: Dict[str, Any]) -> ResponseStatus:
        """
        Map raw response data to ResponseStatus.

        Args:
            response_data: Raw response from exchange

        Returns:
            Mapped response status
        """
        if "error" in response_data or response_data.get("success") is False:
            return ResponseStatus.FAILURE
        elif response_data.get("partial", False):
            return ResponseStatus.PARTIAL_SUCCESS
        else:
            return ResponseStatus.SUCCESS

    async def get_aggregated_response(self, request_id: str) -> Optional[AggregatedResponse]:
        """
        Get aggregated response for a request.

        Args:
            request_id: ID of the request

        Returns:
            Aggregated response if available
        """
        return self.aggregated_responses.get(request_id)

    async def get_pending_responses(self, request_id: str) -> Dict[str, ExchangeResponse]:
        """
        Get pending responses for a request.

        Args:
            request_id: ID of the request

        Returns:
            Pending responses
        """
        return self.pending_responses.get(request_id, {})

    async def clear_responses(self, request_id: str) -> None:
        """
        Clear responses for a request.

        Args:
            request_id: ID of the request to clear
        """
        self.pending_responses.pop(request_id, None)
        self.aggregated_responses.pop(request_id, None)

    async def get_response_statistics(self) -> Dict[str, Any]:
        """Get response handling statistics"""
        total_pending = sum(len(responses) for responses in self.pending_responses.values())
        total_aggregated = len(self.aggregated_responses)
        callback_count = sum(len(callbacks) for callbacks in self.response_callbacks.values())

        return {
            "pending_responses": total_pending,
            "aggregated_responses": total_aggregated,
            "registered_callbacks": callback_count,
            "response_types": list(self.response_callbacks.keys()),
            "running": self._running
        }


class ResponseAggregator:
    """
    Aggregates responses from multiple exchanges based on different strategies.
    """

    def __init__(self):
        self.logger = logging.getLogger("ResponseAggregator")
        self.aggregation_strategies: Dict[str, Callable] = {
            "consensus": self._consensus_aggregation,
            "best_price": self._best_price_aggregation,
            "weighted_average": self._weighted_average_aggregation,
            "first_success": self._first_success_aggregation,
            "all_required": self._all_required_aggregation
        }

    async def aggregate(
        self,
        responses: Dict[str, ExchangeResponse],
        strategy: str = "first_success",
        **strategy_params
    ) -> AggregatedResponse:
        """
        Aggregate responses using the specified strategy.

        Args:
            responses: Responses to aggregate
            strategy: Aggregation strategy
            **strategy_params: Parameters for the strategy

        Returns:
            Aggregated response
        """
        if strategy not in self.aggregation_strategies:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        aggregation_func = self.aggregation_strategies[strategy]
        return await aggregation_func(responses, **strategy_params)

    async def _consensus_aggregation(
        self,
        responses: Dict[str, ExchangeResponse],
        required_consensus: float = 0.8
    ) -> AggregatedResponse:
        """
        Aggregate using consensus strategy.

        Args:
            responses: Responses to aggregate
            required_consensus: Required consensus ratio (0.0 to 1.0)

        Returns:
            Aggregated response
        """
        # Implementation would require specific logic based on response types
        # For now, return a basic aggregation
        return AggregatedResponse(
            response_id=f"consensus_{int(datetime.now().timestamp() * 1000)}",
            request_id="unknown",
            response_type=ResponseType.MARKET_DATA,
            aggregated_data={"strategy": "consensus"},
            individual_responses=responses,
            successful_exchanges=list(responses.keys())
        )

    async def _best_price_aggregation(
        self,
        responses: Dict[str, ExchangeResponse],
        **kwargs
    ) -> AggregatedResponse:
        """
        Aggregate by selecting the best price.

        Args:
            responses: Responses to aggregate

        Returns:
            Aggregated response with best price
        """
        # Implementation would find the best price across exchanges
        # For now, return a basic aggregation
        return AggregatedResponse(
            response_id=f"best_price_{int(datetime.now().timestamp() * 1000)}",
            request_id="unknown",
            response_type=ResponseType.MARKET_DATA,
            aggregated_data={"strategy": "best_price"},
            individual_responses=responses,
            successful_exchanges=list(responses.keys())
        )

    async def _weighted_average_aggregation(
        self,
        responses: Dict[str, ExchangeResponse],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> AggregatedResponse:
        """
        Aggregate using weighted average.

        Args:
            responses: Responses to aggregate
            weights: Weights for each exchange

        Returns:
            Weighted average response
        """
        # Implementation would calculate weighted averages
        # For now, return a basic aggregation
        return AggregatedResponse(
            response_id=f"weighted_avg_{int(datetime.now().timestamp() * 1000)}",
            request_id="unknown",
            response_type=ResponseType.MARKET_DATA,
            aggregated_data={"strategy": "weighted_average"},
            individual_responses=responses,
            successful_exchanges=list(responses.keys())
        )

    async def _first_success_aggregation(
        self,
        responses: Dict[str, ExchangeResponse],
        **kwargs
    ) -> AggregatedResponse:
        """
        Aggregate by taking the first successful response.

        Args:
            responses: Responses to aggregate

        Returns:
            First successful response
        """
        for exchange, response in responses.items():
            if response.status == ResponseStatus.SUCCESS:
                return AggregatedResponse(
                    response_id=f"first_success_{int(datetime.now().timestamp() * 1000)}",
                    request_id="unknown",
                    response_type=response.response_type,
                    aggregated_data=response.data,
                    individual_responses=responses,
                    successful_exchanges=[exchange],
                    failed_exchanges=[ex for ex in responses.keys() if ex != exchange]
                )

        # If no successful responses, return failure
        return AggregatedResponse(
            response_id=f"first_success_fail_{int(datetime.now().timestamp() * 1000)}",
            request_id="unknown",
            response_type=ResponseType.ERROR,
            aggregated_data={},
            individual_responses=responses,
            successful_exchanges=[],
            failed_exchanges=list(responses.keys())
        )

    async def _all_required_aggregation(
        self,
        responses: Dict[str, ExchangeResponse],
        required_exchanges: List[str],
        **kwargs
    ) -> AggregatedResponse:
        """
        Aggregate requiring all specified exchanges to succeed.

        Args:
            responses: Responses to aggregate
            required_exchanges: List of exchanges that must succeed

        Returns:
            Aggregated response
        """
        successful_required = [
            ex for ex in required_exchanges
            if ex in responses and responses[ex].status == ResponseStatus.SUCCESS
        ]

        all_successful = len(successful_required) == len(required_exchanges)

        return AggregatedResponse(
            response_id=f"all_required_{int(datetime.now().timestamp() * 1000)}",
            request_id="unknown",
            response_type=ResponseType.MARKET_DATA,
            aggregated_data={"all_successful": all_successful},
            individual_responses=responses,
            successful_exchanges=successful_required,
            failed_exchanges=[ex for ex in required_exchanges if ex not in successful_required]
        )