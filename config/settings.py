"""
Configuration settings for Quotex Trading Bot
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
HISTORICAL_DIR = DATA_DIR / "historical"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, HISTORICAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Trading Configuration
TRADING_CONFIG = {
    "assets": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY"],
    "timeframes": ["1m", "5m", "15m", "30m", "1h"],
    "primary_timeframe": "5m",
    "trade_amount": 10,  # Starting trade amount
    "max_trade_amount": 100,
    "max_daily_trades": 50,
    "max_concurrent_trades": 3,
    "trading_hours": {
        "start": "08:00",
        "end": "18:00",
        "timezone": "UTC"
    },
    "min_confidence": 0.7,  # Minimum confidence for trade execution
    "auto_trading": True,
    "demo_mode": True,  # Set to False for live trading
}

# Risk Management Configuration
RISK_CONFIG = {
    "max_risk_per_trade": 0.02,  # 2% of account balance
    "max_daily_loss": 0.10,  # 10% of account balance
    "max_drawdown": 0.20,  # 20% maximum drawdown
    "profit_target": 0.05,  # 5% daily profit target
    "stop_loss_multiplier": 2.0,
    "take_profit_multiplier": 3.0,
    "martingale_enabled": False,
    "martingale_multiplier": 2.0,
    "max_martingale_steps": 3,
    "dynamic_position_sizing": True,
    "kelly_criterion": True,
}

# Technical Analysis Configuration
TA_CONFIG = {
    "indicators": {
        "sma": [20, 50, 200],
        "ema": [12, 26, 50],
        "rsi": {"period": 14, "overbought": 70, "oversold": 30},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"period": 20, "std": 2},
        "stochastic": {"k_period": 14, "d_period": 3},
        "adx": {"period": 14, "threshold": 25},
        "williams_r": {"period": 14},
        "cci": {"period": 20},
        "mfi": {"period": 14}
    },
    "custom_indicators": {
        "trend_strength": True,
        "support_resistance": True,
        "volume_profile": True,
        "market_structure": True
    },
    "signal_weights": {
        "trend": 0.35,
        "momentum": 0.25,
        "volume": 0.20,
        "volatility": 0.20
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    "models": {
        "primary": "ensemble",  # ensemble, xgboost, lightgbm, neural_network
        "ensemble_models": ["xgboost", "lightgbm", "neural_network"],
        "retrain_interval": 24,  # hours
        "min_training_samples": 1000,
        "validation_split": 0.2,
        "test_split": 0.1
    },
    "features": {
        "technical_indicators": True,
        "price_action": True,
        "volume_indicators": True,
        "market_microstructure": True,
        "sentiment_analysis": False,  # Requires news API
        "time_features": True,
        "lag_features": [1, 2, 3, 5, 10],
        "rolling_stats": [5, 10, 20, 50]
    },
    "preprocessing": {
        "normalize": True,
        "handle_missing": "forward_fill",
        "outlier_detection": True,
        "feature_selection": True,
        "pca": False
    },
    "hyperparameters": {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "neural_network": {
            "hidden_layers": [128, 64, 32],
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
    }
}

# API Configuration
API_CONFIG = {
    "quotex": {
        "base_url": "https://api.quotex.io",
        "websocket_url": "wss://ws.quotex.io",
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1
    },
    "rate_limits": {
        "requests_per_minute": 100,
        "websocket_connections": 5
    }
}

# Database Configuration
DATABASE_CONFIG = {
    "type": "sqlite",
    "name": "quotex_trading.db",
    "path": DATA_DIR / "quotex_trading.db",
    "backup_interval": 3600,  # seconds
    "max_backup_files": 10,
    "tables": {
        "trades": True,
        "market_data": True,
        "predictions": True,
        "performance": True,
        "signals": True
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    "file_rotation": "1 day",
    "file_retention": "30 days",
    "max_file_size": "10 MB",
    "console_output": True,
    "file_output": True,
    "log_file": LOGS_DIR / "quotex_bot.log"
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    "enabled": True,
    "methods": ["console", "file"],  # Add "email", "telegram", "discord" if needed
    "triggers": {
        "trade_opened": True,
        "trade_closed": True,
        "profit_target_reached": True,
        "stop_loss_hit": True,
        "daily_summary": True,
        "system_errors": True
    }
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "initial_balance": 1000,
    "commission": 0.0,  # Quotex doesn't charge commission
    "slippage": 0.0001,  # Minimal slippage for binary options
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "benchmark": "EURUSD",
    "metrics": ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]
}

# Environment Variables (override with .env file)
def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback to default"""
    return os.getenv(key, default)

# Quotex API credentials (set via environment variables)
QUOTEX_EMAIL = get_env_var("QUOTEX_EMAIL", "")
QUOTEX_PASSWORD = get_env_var("QUOTEX_PASSWORD", "")
QUOTEX_API_KEY = get_env_var("QUOTEX_API_KEY", "")

# Security settings
SECURITY_CONFIG = {
    "encrypt_credentials": True,
    "api_key_rotation": True,
    "secure_logging": True,
    "max_login_attempts": 3,
    "session_timeout": 3600  # seconds
}

# Development/Debug settings
DEBUG_CONFIG = {
    "debug_mode": get_env_var("DEBUG", "False").lower() == "true",
    "verbose_logging": get_env_var("VERBOSE", "False").lower() == "true",
    "save_debug_data": True,
    "plot_charts": False,  # Set to True for visual debugging
    "dry_run": get_env_var("DRY_RUN", "False").lower() == "true"
}