"""
Database package for Quotex Trading Bot

This package contains all database-related functionality including:
- SQLAlchemy models for all data entities
- Database manager for CRUD operations
- Data export/import utilities
- Database maintenance functions
"""

from .models import (
    Base,
    TradingAccount,
    Asset,
    Trade,
    MarketData,
    TradingSignal,
    Strategy,
    BacktestResult,
    MLModel,
    ModelPrediction,
    SystemLog,
    SystemConfig
)

from .database_manager import DatabaseManager

__all__ = [
    'Base',
    'TradingAccount',
    'Asset',
    'Trade',
    'MarketData',
    'TradingSignal',
    'Strategy',
    'BacktestResult',
    'MLModel',
    'ModelPrediction',
    'SystemLog',
    'SystemConfig',
    'DatabaseManager'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Quotex Trading Bot'
__description__ = 'Database layer for trading bot operations'