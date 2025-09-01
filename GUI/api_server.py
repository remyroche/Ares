from src.utils.logger import system_logger
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
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST
from src.utils.prometheus_metrics import metrics as prometheus_metrics
import time

# Setup logger
logger = logging.getLogger(__name__)

# --- Project Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# --- Import from your Ares Codebase ---
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
    from src.config import CONFIG, AresConfig
    from src.database.sqlite_manager import SQLiteManager
    from src.supervisor.performance_reporter import PerformanceReporter
    from src.utils.state_manager import StateManager
    from src.monitoring.metrics_dashboard import MetricsDashboard
    from src.monitoring.performance_dashboard import PerformanceDashboard
    from src.monitoring.enhanced_ml_tracker import EnhancedMLTracker
    from src.monitoring.ml_monitor import MLMonitor
    from src.monitoring.performance_monitor import PerformanceMonitor
    ares_config = AresConfig()
    print("Successfully imported Ares modules.")
except ImportError as e:
    passpasspasspasspasspasspassprint(f"Error importing Ares modules: {e}")
    print(
        "Please ensure the project structure is correct and all dependencies are installed.",
    )

    # Define dummy classes if imports fail
    class SQLiteManager:
    passpassdef __init__(...):
    try:
            # Train the model
            self.model.fit(X_train, y_train, validation_data=(X_val, y_val))
            self.logger.info("Model training completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Model training failed: {{e}}")
            return FalseING", {}).get("model_types", {})
        models = []

        for model_name, config in model_configs.items():
    passif config.get("enabled", False):
    passmodels.append(
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
    passpasspasspasspasspasspasslogger.error(f"Error getting available models: {e}")
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
async def get_model_performance(...):
    pass"""Get performance metrics for all models on a specific token/exchange."""
    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Try to get real performance data from performance_reporter
        performances = []

        # Mock performance data based on performance_reporter.py structure
        models = ["lightgbm", "xgboost", "neural_network"]
        for model in models:
    passperformance = ModelPerformance(
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
    passpasspasspasspasspasspasslogger.error(f"Error getting model performance: {e}")
        return []


@app.post("/api/models/select")
async def select_model_for_token(...):
    pass"""Select a model for a specific token/exchange."""
    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        token_key = f"{request.symbol}_{request.exchange}"

        if token_key in token_configs:
    passtoken_configs[token_key].model_version = request.model_version
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
    passpasspasspasspasspasspasslogger.error(f"Error selecting model: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/models/compare/{symbol}/{exchange}", response_model=ModelComparison)
async def compare_models(...):
    pass"""Compare two models for a specific token/exchange."""
    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Get performance data for both models
        performances = await get_model_performance(symbol, exchange)
        model_a_perf = next((p for p in performances if p.model_id == model_a), None)
        model_b_perf = next((p for p in performances if p.model_id == model_b), None)

        if not model_a_perf or not model_b_perf:
    passpassreturn {"error": "One or both models not found"}

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
    passpasspasspasspasspasspasspasslogger.error(f"Error comparing models: {e}")
        return {"error": str(e)}


@app.get("/api/models/analysis/{symbol}/{exchange}/{model_id}")
async def get_detailed_model_analysis(...):
    pass"""Get detailed analysis for a specific model on a token/exchange."""
    try:
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Get performance data
        performances = await get_model_performance(symbol, exchange)
        performance = next((p for p in performances if p.model_id == model_id), None)

        if not performance:
    passpassreturn {"error": "Model not found"}

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
    passpasspasspasspasspasspasslogger.error(f"Error getting model analysis: {e}")
        return {"error": str(e)}

# --- DriftAlert Pydantic Model ---
class DriftAlertModel(BaseModel):
    model_id: str
    model_type: str
    drift_type: str
    drift_score: float
    threshold: float
    timestamp: str
    features_affected: list[str]
    severity: str
    description: str

@app.get("/api/monitoring/drift-alerts", response_model=list[DriftAlertModel])
async def get_drift_alerts(...):
    passif ml_monitor and hasattr(ml_monitor, 'get_drift_alerts'):
    passalerts = ml_monitor.get_drift_alerts()
        # Convert to DriftAlertModel, ensuring timestamp is str
        return [
            DriftAlertModel(
                model_id=a.model_id,
                model_type=a.model_type,
                drift_type=a.drift_type.value if hasattr(a.drift_type, 'value') else str(a.drift_type),
                drift_score=a.drift_score,
                threshold=a.threshold,
                timestamp=a.timestamp.isoformat() if hasattr(a.timestamp, 'isoformat') else str(a.timestamp),
                features_affected=a.features_affected,
                severity=a.severity,
                description=a.description
            ) for a in alerts
        ]
    return []

@app.get("/api/monitoring/feature-importance/{model_id}")
async def get_feature_importance(...):
    passpassif ml_monitor and hasattr(ml_monitor, 'get_feature_importance_history'):
    passreturn [fi.__dict__ for fi in ml_monitor.get_feature_importance_history(model_id)]
    return []

@app.get("/api/monitoring/online-learning/{model_id}")
async def get_online_learning_metrics(...):
    passpassif ml_monitor and hasattr(ml_monitor, 'get_online_learning_metrics'):
    passmetrics = ml_monitor.get_online_learning_metrics(model_id)
        return metrics.__dict__ if metrics else {}
    return {}

# --- ML Tracker Stats Cache ---
_ml_tracker_stats_cache = {
    'data': None,
    'timestamp': 0
}
_ML_TRACKER_STATS_CACHE_TTL = 5  # seconds

@app.get("/api/monitoring/ml-tracker-stats")
async def get_ml_tracker_stats(...):
    passnow = time.time()
    if (
        _ml_tracker_stats_cache['data'] is not None and
        now - _ml_tracker_stats_cache['timestamp'] < _ML_TRACKER_STATS_CACHE_TTL
    ):
    passreturn _ml_tracker_stats_cache['data']
    if enhanced_ml_tracker and hasattr(enhanced_ml_tracker, 'get_tracking_statistics'):
    passstats = await enhanced_ml_tracker.get_tracking_statistics()
        _ml_tracker_stats_cache['data'] = stats
        _ml_tracker_stats_cache['timestamp'] = now
        return stats
    return {}

# Optionally, keep the old endpoints for backward compatibility, but fetch from the cache
@app.get("/api/monitoring/retraining-recommendations")
async def get_retraining_recommendations(...):
    passpassstats = await get_ml_tracker_stats()
    return stats.get('retraining_recommendations', [])

@app.get("/api/monitoring/model-comparison")
async def get_model_comparison(...):
    passstats = await get_ml_tracker_stats()
    return stats.get('comparison_reports', [])

@app.get("/api/monitoring/regime-performance")
async def get_regime_performance(...):
    passstats = await get_ml_tracker_stats()
    return stats.get('regime_performance', {})


if __name__ == "__main__":
    passimport uvicorn

    print("Starting Ares API server v2.0...")
    print("API documentation will be available at http://localhost:8000/docs")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
    uvicorn.run("api_server:app", host="0_2_3.0", port=port, reload=True)
    def _calculate_confidence(self, prediction):
        """Calculate prediction confidence."""
        try:
            if hasattr(prediction, 'predict_proba'):
                return np.max(prediction.predict_proba())
            elif isinstance(prediction, (list, np.ndarray)):
                return np.max(prediction)
            else:
                return 0.5
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0

