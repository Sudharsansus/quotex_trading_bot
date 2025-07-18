"""
Logging configuration and utilities for the Quotex trading bot.
Provides centralized logging with different levels and output formats.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TradingBotLogger:
    """Custom logger class for the trading bot with multiple handlers and formatters."""
    
    def __init__(self, name: str = "quotex_trading_bot", log_dir: str = "data/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up different handlers for console and file logging."""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for general logs
        general_log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error file handler
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Trading activities handler
        trading_log_file = self.log_dir / f"trading_activities.log"
        trading_handler = logging.handlers.RotatingFileHandler(
            trading_log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=10
        )
        trading_handler.setLevel(logging.INFO)
        trading_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trading_handler.setFormatter(trading_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
        # Create separate logger for trading activities
        self.trading_logger = logging.getLogger(f"{self.name}.trading")
        self.trading_logger.setLevel(logging.INFO)
        self.trading_logger.addHandler(trading_handler)
        self.trading_logger.propagate = False
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance for a specific module."""
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def get_trading_logger(self) -> logging.Logger:
        """Get the specialized trading activities logger."""
        return self.trading_logger


# Global logger instance
_bot_logger = None


def setup_logging(log_level: str = "INFO", log_dir: str = "data/logs") -> TradingBotLogger:
    """
    Set up logging for the entire application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    
    Returns:
        TradingBotLogger instance
    """
    global _bot_logger
    
    if _bot_logger is None:
        _bot_logger = TradingBotLogger(log_dir=log_dir)
        
        # Set log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        
        _bot_logger.logger.setLevel(numeric_level)
    
    return _bot_logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        module_name: Name of the module requesting the logger
    
    Returns:
        Logger instance
    """
    global _bot_logger
    
    if _bot_logger is None:
        _bot_logger = setup_logging()
    
    return _bot_logger.get_logger(module_name)


def get_trading_logger() -> logging.Logger:
    """
    Get the specialized trading activities logger.
    
    Returns:
        Trading logger instance
    """
    global _bot_logger
    
    if _bot_logger is None:
        _bot_logger = setup_logging()
    
    return _bot_logger.get_trading_logger()


def log_trade(action: str, symbol: str, amount: float, price: float, 
              trade_id: Optional[str] = None, profit: Optional[float] = None, **kwargs):
    """
    Log trading activities with structured format.
    
    Args:
        action: Trade action (BUY, SELL, OPEN, CLOSE)
        symbol: Trading symbol
        amount: Trade amount
        price: Entry/exit price
        trade_id: Unique trade identifier
        profit: Profit/loss amount
        **kwargs: Additional trade parameters
    """
    trading_logger = get_trading_logger()
    
    trade_info = {
        'action': action,
        'symbol': symbol,
        'amount': amount,
        'price': price,
        'timestamp': datetime.now().isoformat()
    }
    
    if trade_id:
        trade_info['trade_id'] = trade_id
    if profit is not None:
        trade_info['profit'] = profit
    
    trade_info.update(kwargs)
    
    # Format log message
    log_parts = [f"{k}={v}" for k, v in trade_info.items()]
    log_message = " | ".join(log_parts)
    
    trading_logger.info(log_message)


def log_signal(signal_type: str, symbol: str, confidence: float, 
               indicators: dict, **kwargs):
    """
    Log trading signals with detailed information.
    
    Args:
        signal_type: Type of signal (BUY, SELL, HOLD)
        symbol: Trading symbol
        confidence: Signal confidence (0-1)
        indicators: Technical indicators values
        **kwargs: Additional signal parameters
    """
    logger = get_logger("signals")
    
    signal_info = {
        'signal': signal_type,
        'symbol': symbol,
        'confidence': f"{confidence:.4f}",
        'timestamp': datetime.now().isoformat()
    }
    
    # Add indicator values
    for indicator, value in indicators.items():
        if isinstance(value, float):
            signal_info[indicator] = f"{value:.6f}"
        else:
            signal_info[indicator] = str(value)
    
    signal_info.update(kwargs)
    
    # Format log message
    log_parts = [f"{k}={v}" for k, v in signal_info.items()]
    log_message = " | ".join(log_parts)
    
    logger.info(f"SIGNAL: {log_message}")


class ContextLogger:
    """Context manager for logging function execution time and errors."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, 
                          f"Completed {self.operation} in {duration.total_seconds():.3f}s")
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration.total_seconds():.3f}s: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


def log_execution_time(operation: str, level: int = logging.INFO):
    """
    Decorator to log function execution time.
    
    Args:
        operation: Description of the operation
        level: Logging level
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with ContextLogger(logger, f"{operation} ({func.__name__})", level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging("DEBUG")
    
    # Get loggers
    logger = get_logger("test")
    trading_logger = get_trading_logger()
    
    # Test logging
    logger.info("Testing logger functionality")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test trading logger
    log_trade("BUY", "EURUSD", 100.0, 1.1234, trade_id="T001")
    log_signal("BUY", "EURUSD", 0.85, {"rsi": 30.5, "macd": 0.001})
    
    # Test context logger
    logger = get_logger("context_test")
    with ContextLogger(logger, "test operation"):
        import time
        time.sleep(0.1)
    
    print("Logging test completed. Check data/logs/ directory for log files.")