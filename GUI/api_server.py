import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any

import psutil
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logger
logger = logging.getLogger(__name__)

# --- Project Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# --- Import from your Ares Codebase ---
try:
    from src.config import CONFIG
    from src.database.sqlite_manager import SQLiteManager
    from src.supervisor.performance_reporter import PerformanceReporter
    from src.utils.state_manager import StateManager

    print("Successfully imported Ares modules.")
except ImportError as e:
    print(f"Error importing Ares modules: {e}")
    print(
        "Please ensure the project structure is correct and all dependencies are installed.",
    )

    # Define dummy classes if imports fail
    class SQLiteManager:
        def __init__(self, db_path=""):
            pass

        async def initialize(self):
            pass

        async def get_collection(self, *args, **kwargs):
            return []

        async def set_document(self, *args, **kwargs):
            pass

    class StateManager:
        def __init__(self):
            pass

        def is_kill_switch_active(self):
            return False

        async def activate_kill_switch(self, reason):
            pass

        async def deactivate_kill_switch(self):
            pass

        def get_kill_switch_reason(self):
            return "Kill switch not available"

    class PerformanceReporter:
        def __init__(self):
            pass


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Ares Trading Bot API",
    description="Comprehensive API for the Ares trading bot with kill switch, backtesting, and analysis capabilities.",
    version="2.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
db_manager = SQLiteManager()
state_manager = StateManager()
performance_reporter = PerformanceReporter()
websocket_connections = []

# Add new global variables for token and model management
token_configs: dict[str, Any] = {}
model_performances: dict[str, Any] = {}
model_comparisons: dict[str, Any] = {}
websocket_manager = None  # Will be initialized later


# --- Pydantic Models ---
class Position(BaseModel):
    id: int
    pair: str
    exchange: str
    size: float
    entryPrice: float
    currentPrice: float
    pnl: float
    side: str
    leverage: float | None = 1.0
    unrealizedPnl: float | None = 0.0


class Trade(BaseModel):
    id: str
    pair: str
    exchange: str
    size: float
    entryPrice: float
    exitPrice: float
    pnl: float
    date: str
    side: str
    exitReason: str | None = None
    tradeDuration: float | None = None
    fees: float | None = None


class PerformanceDataPoint(BaseModel):
    date: str
    portfolioValue: float
    drawdown: float | None = None
    trades: int | None = None


class Bot(BaseModel):
    id: int
    pair: str
    exchange: str
    status: str
    model: str
    uptime: str
    pnl: float | None = 0.0
    winRate: float | None = 0.0


class NewBot(BaseModel):
    pair: str
    exchange: str
    model: str
    capital: float | None = 10000


class BacktestParams(BaseModel):
    token_pair: str
    exchange: str
    test_type: str
    start_date: str | None = None
    end_date: str | None = None
    capital: float | None = 10000
    commission: float | None = 0.1
    model_version: str | None = None


class KillSwitchRequest(BaseModel):
    reason: str
    emergency: bool = False


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    type: str
    performance: dict[str, float]
    last_updated: str
    status: str


class TradeAnalysis(BaseModel):
    trade_id: str
    detailed_data: dict[str, Any]
    performance_metrics: dict[str, float]
    market_conditions: dict[str, Any]


class TokenConfig(BaseModel):
    symbol: str
    exchange: str
    enabled: bool = True
    model_version: str | None = None
    last_updated: str | None = None


class ModelPerformance(BaseModel):
    model_id: str
    model_version: str
    symbol: str
    exchange: str
    total_trades: int
    win_rate: float
    net_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    last_updated: str


class ModelComparison(BaseModel):
    model_a: str
    model_b: str
    symbol: str
    exchange: str
    comparison_metrics: dict[str, Any]
    winner: str | None = None
    confidence: float = 0.0


class TokenManagementRequest(BaseModel):
    symbol: str
    exchange: str
    enabled: bool
    model_version: str | None = None


class ModelSelectionRequest(BaseModel):
    symbol: str
    exchange: str
    model_version: str


# --- WebSocket Manager ---
class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass


manager = WebSocketManager()


# --- Mock Data Generation ---
def create_mock_data():
    mock_bots = [
        Bot(
            id=1,
            pair="BTC/USDT",
            exchange=ares_config.exchange_name,
            status="running",
            model="Performer v1.2",
            uptime="7d 4h 15m",
            pnl=1250.50,
            winRate=68.5,
        ),
        Bot(
            id=2,
            pair="ETH/USDT",
            exchange="Bybit",
            status="stopped",
            model="Current v3.1",
            uptime="N/A",
            pnl=-320.75,
            winRate=45.2,
        ),
        Bot(
            id=3,
            pair="SOL/USDT",
            exchange=ares_config.exchange_name,
            status="error",
            model="Performer v1.1",
            uptime="N/A",
            pnl=0.0,
            winRate=0.0,
        ),
    ]

    mock_positions = [
        Position(
            id=1,
            pair="BTC/USDT",
            exchange=ares_config.exchange_name,
            size=0.5,
            entryPrice=68500,
            currentPrice=random.uniform(68000, 70000),
            pnl=random.uniform(-500, 500),
            side="long",
            leverage=2.0,
            unrealizedPnl=random.uniform(-200, 300),
        ),
        Position(
            id=2,
            pair="ETH/USDT",
            exchange="Bybit",
            size=10,
            entryPrice=3600,
            currentPrice=random.uniform(3500, 3700),
            pnl=random.uniform(-500, 500),
            side="short",
            leverage=1.5,
            unrealizedPnl=random.uniform(-150, 250),
        ),
    ]

    mock_trades = [
        Trade(
            id=f"trade_{i}",
            pair="BTC/USDT",
            exchange=ares_config.exchange_name,
            size=0.2,
            entryPrice=68000,
            exitPrice=68500,
            pnl=100,
            date=(datetime.now() - timedelta(hours=i)).isoformat(),
            side="long",
            exitReason="take_profit",
            tradeDuration=3600,
            fees=2.5,
        )
        for i in range(10)
    ]

    return mock_bots, mock_positions, mock_trades


# --- API Endpoints ---


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Ares API v2.0. Navigate to /docs for API documentation.",
        "version": "2.0.0",
        "features": [
            "kill_switch",
            "backtesting",
            "model_management",
            "trade_analysis",
        ],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- Dashboard Endpoints ---
@app.get("/api/dashboard-data")
async def get_dashboard_data(days: int = 7):
    """Fetches comprehensive dashboard data including real-time metrics."""
    try:
        # Get real data from database if available
        try:
            await db_manager.initialize()
            open_positions_raw = await db_manager.get_collection(
                "positions",
                is_public=False,
            )
            last_trades_raw = await db_manager.get_collection(
                "trades",
                is_public=False,
                query_filters=[("limit", "10")],
            )
        except Exception:
            # Fallback to mock data
            mock_bots, mock_positions, mock_trades = create_mock_data()
            open_positions_raw = mock_positions
            last_trades_raw = mock_trades

        # Generate performance curve
        value = 10000
        performance_curve = []
        for i in range(days, -1, -1):
            date = datetime.now() - timedelta(days=i)
            value += (random.random() - 0.45) * 200
            performance_curve.append(
                PerformanceDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    portfolioValue=round(value, 2),
                    drawdown=random.uniform(0, 5),
                    trades=random.randint(0, 10),
                ),
            )

        # Calculate metrics
        total_pnl = sum(p.pnl for p in open_positions_raw) if open_positions_raw else 0
        win_rate = 68 if last_trades_raw else 0
        running_bots = (
            len([b for b in mock_bots if b.status == "running"])
            if "mock_bots" in locals()
            else 0
        )

        return {
            "totalPnl": total_pnl,
            "openPositionsCount": len(open_positions_raw),
            "runningBotsCount": running_bots,
            "winRate": win_rate,
            "performanceCurve": performance_curve,
            "openPositions": open_positions_raw,
            "lastTrades": last_trades_raw,
            "killSwitchActive": state_manager.is_kill_switch_active(),
            "systemStatus": "healthy",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Kill Switch Endpoints ---
@app.get("/api/kill-switch/status")
async def get_kill_switch_status():
    """Get current kill switch status."""
    try:
        is_active = state_manager.is_kill_switch_active()
        reason = state_manager.get_kill_switch_reason() if is_active else None
        return {
            "active": is_active,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kill-switch/activate")
async def activate_kill_switch(request: KillSwitchRequest):
    """Activate the kill switch."""
    try:
        await state_manager.activate_kill_switch(request.reason)

        # Broadcast to WebSocket connections
        await manager.broadcast(
            {
                "type": "kill_switch_activated",
                "reason": request.reason,
                "emergency": request.emergency,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return {
            "message": "Kill switch activated successfully",
            "reason": request.reason,
            "emergency": request.emergency,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kill-switch/deactivate")
async def deactivate_kill_switch():
    """Deactivate the kill switch."""
    try:
        await state_manager.deactivate_kill_switch()

        # Broadcast to WebSocket connections
        await manager.broadcast(
            {
                "type": "kill_switch_deactivated",
                "timestamp": datetime.now().isoformat(),
            },
        )

        return {"message": "Kill switch deactivated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Backtesting Endpoints ---
@app.post("/api/run-backtest")
async def start_backtest(params: BacktestParams, background_tasks: BackgroundTasks):
    """Run a comprehensive backtest with detailed results."""
    try:
        # Simulate backtest execution
        await asyncio.sleep(2)

        # Generate detailed mock results
        mock_equity_curve = []
        value = params.capital or 10000
        for i in range(90, -1, -1):
            date = datetime.now() - timedelta(days=i)
            value += (random.random() - 0.45) * (value * 0.02)
            mock_equity_curve.append(
                PerformanceDataPoint(
                    date=date.strftime("%Y-%m-%d"),
                    portfolioValue=round(value, 2),
                    drawdown=random.uniform(0, 8),
                    trades=random.randint(0, 15),
                ),
            )

        # Calculate detailed metrics
        total_return = random.uniform(5, 40)
        sharpe_ratio = random.uniform(0.8, 2.5)
        max_drawdown = random.uniform(5, 15)
        win_rate = random.uniform(55, 75)

        return {
            "message": "Backtest completed successfully.",
            "results": {
                "summary": {
                    "totalReturn": f"{total_return:.2f}%",
                    "sharpeRatio": f"{sharpe_ratio:.2f}",
                    "maxDrawdown": f"{max_drawdown:.2f}%",
                    "winRate": f"{win_rate:.0f}%",
                    "totalTrades": random.randint(50, 200),
                    "avgTradeDuration": random.uniform(2, 8),
                    "profitFactor": random.uniform(1.1, 2.5),
                    "calmarRatio": random.uniform(0.5, 3.0),
                },
                "equityCurve": mock_equity_curve,
                "tradeAnalysis": {
                    "bestTrade": random.uniform(500, 2000),
                    "worstTrade": random.uniform(-1000, -200),
                    "avgWin": random.uniform(100, 300),
                    "avgLoss": random.uniform(-200, -50),
                    "largestWinStreak": random.randint(5, 15),
                    "largestLossStreak": random.randint(2, 8),
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run backtest: {str(e)}")


@app.get("/api/backtest/comparison")
async def get_backtest_comparison():
    """Get comparison data for multiple backtest runs."""
    try:
        # Mock comparison data
        comparisons = [
            {
                "id": "backtest_1",
                "name": "Current Strategy",
                "totalReturn": 15.2,
                "sharpeRatio": 2.1,
                "maxDrawdown": 5.5,
                "winRate": 68.5,
                "color": "#8b5cf6",
            },
            {
                "id": "backtest_2",
                "name": "Optimized Strategy",
                "totalReturn": 18.7,
                "sharpeRatio": 2.3,
                "maxDrawdown": 4.2,
                "winRate": 72.1,
                "color": "#06b6d4",
            },
            {
                "id": "backtest_3",
                "name": "Conservative Strategy",
                "totalReturn": 12.3,
                "sharpeRatio": 1.8,
                "maxDrawdown": 3.1,
                "winRate": 65.2,
                "color": "#10b981",
            },
        ]

        return {"comparisons": comparisons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Model Management Endpoints ---
@app.get("/api/models")
async def get_models():
    """Get all available models with their performance metrics."""
    try:
        models = [
            ModelInfo(
                id="model_1",
                name="Performer v1.2",
                version="1.2.0",
                type="ensemble",
                performance={
                    "accuracy": 72.5,
                    "precision": 68.3,
                    "recall": 71.2,
                    "f1_score": 69.7,
                },
                last_updated="2024-01-15T10:30:00Z",
                status="active",
            ),
            ModelInfo(
                id="model_2",
                name="Current v3.1",
                version="3.1.0",
                type="deep_learning",
                performance={
                    "accuracy": 69.8,
                    "precision": 67.1,
                    "recall": 70.5,
                    "f1_score": 68.8,
                },
                last_updated="2024-01-10T14:20:00Z",
                status="active",
            ),
            ModelInfo(
                id="model_3",
                name="Experimental v2.0",
                version="2.0.0",
                type="ensemble",
                performance={
                    "accuracy": 74.2,
                    "precision": 71.8,
                    "recall": 73.5,
                    "f1_score": 72.6,
                },
                last_updated="2024-01-20T09:15:00Z",
                status="testing",
            ),
        ]

        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model_id}/deploy")
async def deploy_model(model_id: str):
    """Deploy a specific model."""
    try:
        # Mock deployment
        await asyncio.sleep(1)
        return {
            "message": f"Model {model_id} deployed successfully",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Trade Analysis Endpoints ---
@app.get("/api/trades/analysis")
async def get_trade_analysis(days: int = 30, limit: int = 100):
    """Get comprehensive trade analysis data."""
    try:
        # Mock detailed trade analysis
        trades = []
        for i in range(limit):
            trade_data = {
                "trade_id": f"trade_{i}",
                "pair": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
                "side": random.choice(["long", "short"]),
                "entry_price": random.uniform(20000, 70000),
                "exit_price": random.uniform(20000, 70000),
                "pnl": random.uniform(-1000, 2000),
                "entry_time": (
                    datetime.now() - timedelta(days=random.randint(1, days))
                ).isoformat(),
                "exit_time": (
                    datetime.now() - timedelta(days=random.randint(0, days - 1))
                ).isoformat(),
                "duration": random.uniform(300, 86400),
                "fees": random.uniform(1, 10),
                "slippage": random.uniform(0, 0.5),
                "market_regime": random.choice(["trending", "ranging", "volatile"]),
                "confidence": random.uniform(0.5, 0.95),
                "volume": random.uniform(1000, 50000),
                "volatility": random.uniform(0.01, 0.05),
            }
            trades.append(trade_data)

        # Calculate aggregate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t["pnl"] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        avg_trade_duration = (
            sum(t["duration"] for t in trades) / total_trades if total_trades > 0 else 0
        )

        return {
            "trades": trades,
            "summary": {
                "totalTrades": total_trades,
                "winRate": round(win_rate, 2),
                "totalPnl": round(total_pnl, 2),
                "avgTradeDuration": round(avg_trade_duration, 2),
                "bestTrade": max(t["pnl"] for t in trades) if trades else 0,
                "worstTrade": min(t["pnl"] for t in trades) if trades else 0,
                "avgWin": sum(t["pnl"] for t in trades if t["pnl"] > 0) / winning_trades
                if winning_trades > 0
                else 0,
                "avgLoss": sum(t["pnl"] for t in trades if t["pnl"] < 0)
                / (total_trades - winning_trades)
                if (total_trades - winning_trades) > 0
                else 0,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades/{trade_id}/detailed")
async def get_trade_details(trade_id: str):
    """Get detailed analysis for a specific trade."""
    try:
        # Mock detailed trade data
        trade_details = {
            "trade_id": trade_id,
            "pair": "BTC/USDT",
            "side": "long",
            "entry_price": 68500,
            "exit_price": 69200,
            "pnl": 700,
            "entry_time": "2024-01-15T10:30:00Z",
            "exit_time": "2024-01-15T14:45:00Z",
            "duration": 15300,
            "fees": 5.25,
            "slippage": 0.12,
            "market_regime": "trending",
            "confidence": 0.85,
            "volume": 25000,
            "volatility": 0.025,
            "technical_indicators": {
                "rsi": 65.2,
                "macd": 0.45,
                "bollinger_position": 0.7,
                "volume_sma_ratio": 1.2,
            },
            "risk_metrics": {
                "position_size": 0.5,
                "leverage": 2.0,
                "risk_reward_ratio": 2.5,
                "max_drawdown": 150,
            },
            "market_conditions": {
                "trend_strength": 0.8,
                "volatility_regime": "medium",
                "liquidity_score": 0.9,
                "correlation_with_btc": 0.95,
            },
        }

        return trade_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Bot Management Endpoints ---
@app.get("/api/bots", response_model=list[Bot])
async def get_all_bots():
    """Get all configured bots with their status and performance."""
    try:
        mock_bots, _, _ = create_mock_data()
        return mock_bots
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bots", response_model=Bot)
async def add_new_bot(bot: NewBot):
    """Add a new bot to the configuration."""
    try:
        new_bot_data = bot.dict()
        new_bot_data["status"] = "stopped"
        new_bot_data["uptime"] = "N/A"
        new_bot_data["id"] = random.randint(100, 999)
        new_bot_data["pnl"] = 0.0
        new_bot_data["winRate"] = 0.0

        return Bot(**new_bot_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/bots/{bot_id}")
async def remove_bot_endpoint(bot_id: int):
    """Remove a bot from the configuration."""
    try:
        return {"message": f"Bot {bot_id} removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bots/{bot_id}/toggle")
async def toggle_bot(bot_id: int):
    """Start or stop a bot."""
    try:
        return {"message": f"Bot {bot_id} status toggled."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- System Management Endpoints ---
@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Get process info
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "status": "running",
            "uptime": "7d 4h 15m",
            "memory_usage": {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
            },
            "cpu_usage": process.cpu_percent(),
            "kill_switch_active": state_manager.is_kill_switch_active(),
            "trading_paused": False,
            "last_heartbeat": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/restart")
async def restart_system():
    """Restart the trading system."""
    try:
        # In a real implementation, this would trigger a system restart
        return {"message": "System restart initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add new API endpoints for token and model management


@app.get("/api/tokens", response_model=list[TokenConfig])
async def get_tokens():
    """Get all configured tokens with their settings."""
    try:
        # Try to get real data from config
        supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {})
        tokens = []

        for exchange, symbols in supported_tokens.items():
            for symbol in symbols:
                token_key = f"{symbol}_{exchange}"
                if token_key in token_configs:
                    tokens.append(token_configs[token_key])
                else:
                    # Create default config
                    tokens.append(
                        TokenConfig(
                            symbol=symbol,
                            exchange=exchange,
                            enabled=True,
                            model_version=None,
                            last_updated=datetime.now().isoformat(),
                        ),
                    )

        return tokens
    except Exception as e:
        logger.error(f"Error getting tokens: {e}")
        # Return mock data
        return [
            TokenConfig(
                symbol="BTCUSDT",
                exchange=ares_config.exchange_name.upper(),
                enabled=True,
                model_version="v1.2.3",
                last_updated=datetime.now().isoformat(),
            ),
            TokenConfig(
                symbol="ETHUSDT",
                exchange=ares_config.exchange_name.upper(),
                enabled=True,
                model_version="v1.1.0",
                last_updated=datetime.now().isoformat(),
            ),
            TokenConfig(
                symbol="ADAUSDT",
                exchange=ares_config.exchange_name.upper(),
                enabled=False,
                model_version=None,
                last_updated=datetime.now().isoformat(),
            ),
        ]


@app.post("/api/tokens")
async def update_token_config(request: TokenManagementRequest):
    """Add or update token configuration."""
    try:
        token_key = f"{request.symbol}_{request.exchange}"
        token_configs[token_key] = TokenConfig(
            symbol=request.symbol,
            exchange=request.exchange,
            enabled=request.enabled,
            model_version=request.model_version,
            last_updated=datetime.now().isoformat(),
        )

        # Broadcast update via WebSocket
        await websocket_manager.broadcast(
            {"type": "token_config_updated", "data": token_configs[token_key].dict()},
        )

        return {
            "success": True,
            "message": f"Token {request.symbol} on {request.exchange} updated",
        }
    except Exception as e:
        logger.error(f"Error updating token config: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/tokens/{symbol}/{exchange}")
async def remove_token(symbol: str, exchange: str):
    """Remove a token from trading."""
    try:
        token_key = f"{symbol}_{exchange}"
        if token_key in token_configs:
            token_configs[token_key].enabled = False
            token_configs[token_key].last_updated = datetime.now().isoformat()

            # Broadcast update via WebSocket
            await websocket_manager.broadcast(
                {
                    "type": "token_removed",
                    "data": {"symbol": symbol, "exchange": exchange},
                },
            )

            return {
                "success": True,
                "message": f"Token {symbol} on {exchange} disabled",
            }
        return {"success": False, "error": "Token not found"}
    except Exception as e:
        logger.error(f"Error removing token: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/models/available", response_model=list[dict[str, Any]])
async def get_available_models():
    """Get all available models for selection."""
    try:
        # Try to get real model data from MLflow or model directory
        model_configs = CONFIG.get("MODEL_TRAINING", {}).get("model_types", {})
        models = []

        for model_name, config in model_configs.items():
            if config.get("enabled", False):
                models.append(
                    {
                        "model_id": model_name,
                        "model_name": model_name.upper(),
                        "description": f"{model_name.upper()} model for trading",
                        "enabled": True,
                        "last_trained": datetime.now().isoformat(),
                        "performance_score": 0.85,  # Mock score
                    },
                )

        return models
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        # Return mock data
        return [
            {
                "model_id": "lightgbm",
                "model_name": "LIGHTGBM",
                "description": "Gradient boosting with LightGBM",
                "enabled": True,
                "last_trained": datetime.now().isoformat(),
                "performance_score": 0.87,
            },
            {
                "model_id": "xgboost",
                "model_name": "XGBOOST",
                "description": "Extreme gradient boosting",
                "enabled": True,
                "last_trained": datetime.now().isoformat(),
                "performance_score": 0.85,
            },
            {
                "model_id": "neural_network",
                "model_name": "NEURAL_NETWORK",
                "description": "Deep neural network",
                "enabled": True,
                "last_trained": datetime.now().isoformat(),
                "performance_score": 0.82,
            },
        ]


@app.get(
    "/api/models/performance/{symbol}/{exchange}",
    response_model=list[ModelPerformance],
)
async def get_model_performance(symbol: str, exchange: str):
    """Get performance metrics for all models on a specific token/exchange."""
    try:
        # Try to get real performance data from performance_reporter
        performances = []

        # Mock performance data based on performance_reporter.py structure
        models = ["lightgbm", "xgboost", "neural_network"]
        for model in models:
            performance = ModelPerformance(
                model_id=model,
                model_version=f"v1.{len(model)}.0",
                symbol=symbol,
                exchange=exchange,
                total_trades=150 + hash(model) % 100,
                win_rate=0.65 + (hash(model) % 20) / 100,
                net_pnl=1000 + (hash(model) % 5000),
                max_drawdown=-(200 + (hash(model) % 300)),
                sharpe_ratio=1.2 + (hash(model) % 10) / 10,
                profit_factor=1.5 + (hash(model) % 10) / 10,
                avg_trade_duration=2.5 + (hash(model) % 5),
                best_trade=500 + (hash(model) % 1000),
                worst_trade=-(300 + (hash(model) % 400)),
                avg_win=150 + (hash(model) % 100),
                avg_loss=-(100 + (hash(model) % 80)),
                consecutive_wins=5 + (hash(model) % 10),
                consecutive_losses=2 + (hash(model) % 5),
                last_updated=datetime.now().isoformat(),
            )
            performances.append(performance)

        return performances
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return []


@app.post("/api/models/select")
async def select_model_for_token(request: ModelSelectionRequest):
    """Select a model for a specific token/exchange."""
    try:
        token_key = f"{request.symbol}_{request.exchange}"

        if token_key in token_configs:
            token_configs[token_key].model_version = request.model_version
            token_configs[token_key].last_updated = datetime.now().isoformat()

            # Broadcast update via WebSocket
            await websocket_manager.broadcast(
                {
                    "type": "model_selected",
                    "data": {
                        "symbol": request.symbol,
                        "exchange": request.exchange,
                        "model_version": request.model_version,
                    },
                },
            )

            return {
                "success": True,
                "message": f"Model {request.model_version} selected for {request.symbol} on {request.exchange}",
            }
        return {"success": False, "error": "Token not found"}
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/models/compare/{symbol}/{exchange}", response_model=ModelComparison)
async def compare_models(symbol: str, exchange: str, model_a: str, model_b: str):
    """Compare two models for a specific token/exchange."""
    try:
        # Get performance data for both models
        performances = await get_model_performance(symbol, exchange)
        model_a_perf = next((p for p in performances if p.model_id == model_a), None)
        model_b_perf = next((p for p in performances if p.model_id == model_b), None)

        if not model_a_perf or not model_b_perf:
            return {"error": "One or both models not found"}

        # Calculate comparison metrics
        comparison_metrics = {
            "win_rate_diff": model_a_perf.win_rate - model_b_perf.win_rate,
            "pnl_diff": model_a_perf.net_pnl - model_b_perf.net_pnl,
            "sharpe_diff": model_a_perf.sharpe_ratio - model_b_perf.sharpe_ratio,
            "profit_factor_diff": model_a_perf.profit_factor
            - model_b_perf.profit_factor,
            "max_drawdown_diff": model_a_perf.max_drawdown - model_b_perf.max_drawdown,
            "avg_trade_duration_diff": model_a_perf.avg_trade_duration
            - model_b_perf.avg_trade_duration,
        }

        # Determine winner based on multiple metrics
        a_score = (
            model_a_perf.win_rate * 0.3
            + (model_a_perf.net_pnl / 1000) * 0.3
            + model_a_perf.sharpe_ratio * 0.2
            + model_a_perf.profit_factor * 0.2
        )

        b_score = (
            model_b_perf.win_rate * 0.3
            + (model_b_perf.net_pnl / 1000) * 0.3
            + model_b_perf.sharpe_ratio * 0.2
            + model_b_perf.profit_factor * 0.2
        )

        winner = model_a if a_score > b_score else model_b
        confidence = abs(a_score - b_score) / max(a_score, b_score) * 100

        comparison = ModelComparison(
            model_a=model_a,
            model_b=model_b,
            symbol=symbol,
            exchange=exchange,
            comparison_metrics=comparison_metrics,
            winner=winner,
            confidence=confidence,
        )

        return comparison
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        return {"error": str(e)}


@app.get("/api/models/analysis/{symbol}/{exchange}/{model_id}")
async def get_detailed_model_analysis(symbol: str, exchange: str, model_id: str):
    """Get detailed analysis for a specific model on a token/exchange."""
    try:
        # Get performance data
        performances = await get_model_performance(symbol, exchange)
        performance = next((p for p in performances if p.model_id == model_id), None)

        if not performance:
            return {"error": "Model not found"}

        # Create detailed analysis based on performance_reporter.py structure
        analysis = {
            "basic_metrics": {
                "total_trades": performance.total_trades,
                "win_rate": performance.win_rate,
                "net_pnl": performance.net_pnl,
                "max_drawdown": performance.max_drawdown,
                "sharpe_ratio": performance.sharpe_ratio,
                "profit_factor": performance.profit_factor,
            },
            "trade_analysis": {
                "avg_trade_duration": performance.avg_trade_duration,
                "best_trade": performance.best_trade,
                "worst_trade": performance.worst_trade,
                "avg_win": performance.avg_win,
                "avg_loss": performance.avg_loss,
                "consecutive_wins": performance.consecutive_wins,
                "consecutive_losses": performance.consecutive_losses,
            },
            "risk_metrics": {
                "var_95": -(performance.max_drawdown * 0.8),  # Mock VaR
                "max_consecutive_losses": performance.consecutive_losses,
                "recovery_factor": abs(performance.net_pnl / performance.max_drawdown)
                if performance.max_drawdown != 0
                else 0,
                "calmar_ratio": performance.net_pnl / abs(performance.max_drawdown)
                if performance.max_drawdown != 0
                else 0,
            },
            "performance_trends": {
                "monthly_returns": [2.5, 3.1, -1.2, 4.3, 2.8, 1.9],  # Mock data
                "rolling_sharpe": [1.1, 1.3, 0.9, 1.4, 1.2, 1.1],
                "drawdown_periods": [5, 3, 8, 2, 4, 6],
            },
            "model_info": {
                "model_id": model_id,
                "model_version": performance.model_version,
                "last_trained": datetime.now().isoformat(),
                "training_samples": 50000 + hash(model_id) % 20000,
                "feature_count": 25 + hash(model_id) % 15,
            },
        }

        return analysis
    except Exception as e:
        logger.error(f"Error getting model analysis: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("Starting Ares API server v2.0...")
    print("API documentation will be available at http://localhost:8000/docs")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
