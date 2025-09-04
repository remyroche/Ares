#!/usr/bin/env python3
"""
Simplified Ares Trading Bot API Server

This is a standalone version that doesn't require the full Ares codebase
to be available, making it easier to test and run the GUI.
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# Setup logger
logger = logging.getLogger(__name__)

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
websocket_connections = []
kill_switch_active = False
kill_switch_reason = ""

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
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove disconnected connections
                self.disconnect(connection)

manager = WebSocketManager()

# --- Mock Data Generation ---
def create_mock_data():
    mock_bots = [
        Bot(
            id=1,
            pair="BTC/USDT",
            exchange="BINANCE",
            status="running",
            model="Performer v1.2",
            uptime="7d 4h 15m",
            pnl=1250.50,
            winRate=68.5,
        ),
        Bot(
            id=2,
            pair="ETH/USDT",
            exchange="BINANCE",
            status="stopped",
            model="Current v3.1",
            uptime="N/A",
            pnl=-320.75,
            winRate=45.2,
        ),
        Bot(
            id=3,
            pair="SOL/USDT",
            exchange="BINANCE",
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
            exchange="BINANCE",
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
            exchange="BINANCE",
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
            exchange="BINANCE",
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
        # Generate mock data
        mock_bots, mock_positions, mock_trades = create_mock_data()

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
        total_pnl = sum(p.pnl for p in mock_positions)
        win_rate = 68
        running_bots = len([b for b in mock_bots if b.status == "running"])

        return {
            "totalPnl": total_pnl,
            "openPositionsCount": len(mock_positions),
            "runningBotsCount": running_bots,
            "winRate": win_rate,
            "performanceCurve": performance_curve,
            "openPositions": mock_positions,
            "lastTrades": mock_trades,
            "killSwitchActive": kill_switch_active,
            "systemStatus": "healthy",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Kill Switch Endpoints ---
@app.get("/api/kill-switch/status")
async def get_kill_switch_status():
    """Get current kill switch status."""
    try:
        return {
            "active": kill_switch_active,
            "reason": kill_switch_reason if kill_switch_active else None,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kill-switch/activate")
async def activate_kill_switch(request: KillSwitchRequest):
    """Activate the kill switch."""
    try:
        global kill_switch_active, kill_switch_reason
        kill_switch_active = True
        kill_switch_reason = request.reason

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
        global kill_switch_active, kill_switch_reason
        kill_switch_active = False
        kill_switch_reason = ""

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

# --- System Status ---
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
            "kill_switch_active": kill_switch_active,
            "trading_paused": False,
            "last_heartbeat": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Launcher Integration ---
try:
    import sys
    import os
    gui_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, gui_dir)
    from launcher_integration import (
        start_launcher_mode, start_training, stop_process, stop_all_processes,
        get_process_status, get_available_modes, get_available_training_modes,
        get_available_exchanges
    )
    LAUNCHER_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Launcher integration not available: {e}")
    LAUNCHER_INTEGRATION_AVAILABLE = False

@app.get("/api/launcher/status")
async def get_launcher_status():
    """Get the current status of the Ares launcher and running processes."""
    try:
        if LAUNCHER_INTEGRATION_AVAILABLE:
            # Use launcher integration
            status = await get_process_status()
            return {
                "launcher_active": status["total_processes"] > 0,
                "running_processes": status["running_processes"],
                "last_check": status["last_check"],
                "gui_mode": True,
                "integration_available": True
            }
        else:
            # Fallback to process scanning
            running_processes = []
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'ares_launcher.py' in cmdline or 'ares_pipeline.py' in cmdline:
                            running_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline,
                                'status': 'running'
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception:
                pass

            return {
                "launcher_active": len(running_processes) > 0,
                "running_processes": running_processes,
                "last_check": datetime.now().isoformat(),
                "gui_mode": True,
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/launcher/start")
async def start_launcher_mode_endpoint(request: Request):
    """Start a specific launcher mode via the GUI."""
    try:
        data = await request.json()
        mode = data.get('mode')
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'BINANCE')
        lookback_days = data.get('lookback_days')
        
        if not mode or not symbol:
            raise HTTPException(status_code=400, detail="Mode and symbol are required")
        
        if LAUNCHER_INTEGRATION_AVAILABLE:
            # Use launcher integration
            valid_modes = get_available_modes()
            if mode not in valid_modes:
                raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
            result = await start_launcher_mode(mode, symbol, exchange, lookback_days=lookback_days)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": result["message"],
                    "mode": mode,
                    "symbol": symbol,
                    "exchange": exchange,
                    "process_key": result.get("process_key"),
                    "pid": result.get("pid"),
                    "command": result.get("command"),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        else:
            # Fallback to mock response
            valid_modes = ['paper', 'live', 'backtest', 'blank', 'light', 'full', 'load', 'precompute']
            if mode not in valid_modes:
                raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
            return {
                "success": True,
                "message": f"Launcher mode '{mode}' started for {symbol} on {exchange} (mock mode - integration not available)",
                "mode": mode,
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat(),
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/launcher/stop")
async def stop_launcher_endpoint():
    """Stop all running launcher processes."""
    try:
        if LAUNCHER_INTEGRATION_AVAILABLE:
            result = await stop_all_processes()
            return {
                "success": result["success"],
                "message": result["message"],
                "stopped_processes": result.get("stopped_processes", []),
                "errors": result.get("errors", []),
                "timestamp": datetime.now().isoformat(),
                "integration_available": True
            }
        else:
            return {
                "success": True,
                "message": "All launcher processes stopped (mock mode - integration not available)",
                "timestamp": datetime.now().isoformat(),
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Training Endpoints ---
@app.get("/api/training/modes")
async def get_training_modes():
    """Get available training modes and their configurations."""
    try:
        if LAUNCHER_INTEGRATION_AVAILABLE:
            # Use launcher integration
            valid_modes = get_available_training_modes()
            return {
                "modes": {mode: {"description": f"{mode} training mode"} for mode in valid_modes},
                "integration_available": True
            }
        else:
            # Fallback to mock data
            return {
                "modes": {
                    "light": {
                        "description": "Light training mode for quick testing (30 days)",
                        "lookback_days": 30,
                        "max_trials": 10,
                        "n_trials": 5,
                        "computational_intensity": "low",
                        "estimated_duration_minutes": 15,
                        "enable_advanced_model_training": False,
                        "enable_ensemble_training": False,
                        "enable_multi_timeframe_training": False,
                        "enable_adaptive_training": False,
                        "recommendation": "Use for quick testing and development"
                    },
                    "blank": {
                        "description": "Blank training mode for standard testing (180 days)",
                        "lookback_days": 180,
                        "max_trials": 50,
                        "n_trials": 25,
                        "computational_intensity": "medium",
                        "estimated_duration_minutes": 60,
                        "enable_advanced_model_training": True,
                        "enable_ensemble_training": False,
                        "enable_multi_timeframe_training": False,
                        "enable_adaptive_training": False,
                        "recommendation": "Use for standard model training and validation"
                    },
                    "full": {
                        "description": "Full training mode for production (730 days)",
                        "lookback_days": 730,
                        "max_trials": 200,
                        "n_trials": 100,
                        "computational_intensity": "high",
                        "estimated_duration_minutes": 240,
                        "enable_advanced_model_training": True,
                        "enable_ensemble_training": True,
                        "enable_multi_timeframe_training": True,
                        "enable_adaptive_training": True,
                        "recommendation": "Use for production model training with full dataset"
                    }
                },
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/start")
async def start_training_endpoint(request: Request):
    """Start training with specified parameters."""
    try:
        data = await request.json()
        mode = data.get('mode', 'blank')
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'BINANCE')
        lookback_days = data.get('lookback_days')
        
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
        
        if LAUNCHER_INTEGRATION_AVAILABLE:
            # Use launcher integration
            valid_modes = get_available_training_modes()
            if mode not in valid_modes:
                raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
            result = await start_training(mode, symbol, exchange, lookback_days=lookback_days)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": result["message"],
                    "mode": mode,
                    "symbol": symbol,
                    "exchange": exchange,
                    "lookback_days": lookback_days,
                    "process_key": result.get("process_key"),
                    "pid": result.get("pid"),
                    "command": result.get("command"),
                    "timestamp": datetime.now().isoformat(),
                    "integration_available": True
                }
            else:
                raise HTTPException(status_code=500, detail=result["error"])
        else:
            # Fallback to mock response
            valid_modes = ['light', 'blank', 'full']
            if mode not in valid_modes:
                raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
            
            return {
                "success": True,
                "message": f"Training started in {mode} mode for {symbol} on {exchange} (mock mode - integration not available)",
                "mode": mode,
                "symbol": symbol,
                "exchange": exchange,
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat(),
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status and progress."""
    try:
        if LAUNCHER_INTEGRATION_AVAILABLE:
            # Use launcher integration
            status = await get_process_status()
            training_processes = [
                proc for proc in status["running_processes"] 
                if any(mode in proc.get("process_key", "") for mode in ["light", "blank", "full"])
            ]
            
            return {
                "training_active": len(training_processes) > 0,
                "training_processes": training_processes,
                "last_check": status["last_check"],
                "integration_available": True
            }
        else:
            # Fallback to process scanning
            training_processes = []
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'training' in cmdline.lower() or 'enhanced_training_manager' in cmdline:
                            training_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline,
                                'status': 'running'
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception:
                pass
            
            return {
                "training_active": len(training_processes) > 0,
                "training_processes": training_processes,
                "last_check": datetime.now().isoformat(),
                "integration_available": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Data Status ---
@app.get("/api/data/status")
async def get_data_status():
    """Get data collection and processing status."""
    try:
        # Check for data files
        data_files = []
        data_dir = "data_cache"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.parquet') or file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    stat = os.stat(file_path)
                    data_files.append({
                        'name': file,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': 'parquet' if file.endswith('.parquet') else 'csv'
                    })
        
        return {
            "data_files": data_files,
            "data_directory": data_dir,
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Token Management ---
token_configs: Dict[str, TokenConfig] = {}

@app.get("/api/tokens", response_model=List[TokenConfig])
async def get_tokens():
    """Get all configured tokens with their settings."""
    try:
        # Return mock data
        return [
            TokenConfig(
                symbol="BTCUSDT",
                exchange="BINANCE",
                enabled=True,
                model_version="v1.2.3",
                last_updated=datetime.now().isoformat(),
            ),
            TokenConfig(
                symbol="ETHUSDT",
                exchange="BINANCE",
                enabled=True,
                model_version="v1.1.0",
                last_updated=datetime.now().isoformat(),
            ),
            TokenConfig(
                symbol="ADAUSDT",
                exchange="BINANCE",
                enabled=False,
                model_version=None,
                last_updated=datetime.now().isoformat(),
            ),
        ]
    except Exception as e:
        logger.exception(f"Error getting tokens: {e}")
        return []

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
        await manager.broadcast(
            {"type": "token_config_updated", "data": token_configs[token_key].dict()},
        )

        return {
            "success": True,
            "message": f"Token {request.symbol} on {request.exchange} updated",
        }
    except Exception as e:
        logger.exception(f"Error updating token config: {e}")
        return {"success": False, "error": str(e)}

# --- Model Management ---
@app.get("/api/models/available", response_model=List[dict])
async def get_available_models():
    """Get all available models for selection."""
    try:
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
    except Exception as e:
        logger.exception(f"Error getting available models: {e}")
        return []

# --- Backtesting ---
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

# --- Monitoring ---
@app.get("/api/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data."""
    try:
        return {
            "system_health": {
                "status": "healthy",
                "uptime": "7d 4h 15m",
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
            },
            "trading_metrics": {
                "active_bots": 3,
                "total_trades": 150,
                "win_rate": 68.5,
                "total_pnl": 1250.50,
            },
            "model_performance": {
                "accuracy": 0.87,
                "precision": 0.85,
                "recall": 0.82,
                "f1_score": 0.83,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    print("Starting Ares API server v2.0 (Simplified)...")
    print("API documentation will be available at http://localhost:8000/docs")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
    uvicorn.run("api_server_simple:app", host="0.0.0.0", port=port, reload=True)