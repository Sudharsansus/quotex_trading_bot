"""
General utility functions and helpers for the Quotex trading bot.
Contains common functions used across different modules.
"""

import json
import os
import time
import hashlib
import pickle
import gzip
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import functools
import threading
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from urllib.parse import urljoin, urlparse

from .logger import get_logger

logger = get_logger(__name__)


# === Time and Date Utilities ===

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def timestamp_to_datetime(timestamp: Union[int, float, str]) -> datetime:
    """
    Convert timestamp to datetime object.
    
    Args:
        timestamp: Timestamp in various formats (unix, string, etc.)
    
    Returns:
        datetime object
    """
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, (int, float)):
        # Assume unix timestamp
        if timestamp > 1e10:  # Milliseconds
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to unix timestamp."""
    return int(dt.timestamp())


def is_market_open(current_time: Optional[datetime] = None, 
                  market_timezone: str = "America/New_York") -> bool:
    """
    Check if market is open (basic implementation for forex/crypto).
    
    Args:
        current_time: Time to check (default: current time)
        market_timezone: Market timezone
    
    Returns:
        True if market is considered open
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    # Forex market is open 24/5 (closed on weekends)
    weekday = current_time.weekday()
    
    # Monday 00:00 UTC to Friday 23:59 UTC (simplified)
    if weekday < 5:  # Monday to Friday
        return True
    elif weekday == 6 and current_time.hour >= 22:  # Sunday after 22:00 UTC
        return True
    else:
        return False


def get_next_market_open(current_time: Optional[datetime] = None) -> datetime:
    """Get the next market opening time."""
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    if is_market_open(current_time):
        return current_time
    
    # Find next Monday 00:00 UTC
    days_until_monday = (7 - current_time.weekday()) % 7
    if days_until_monday == 0:  # It's Sunday
        days_until_monday = 1
    
    next_open = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    next_open += timedelta(days=days_until_monday)
    
    return next_open


def sleep_until(target_time: datetime):
    """Sleep until a specific datetime."""
    now = datetime.now(timezone.utc)
    if target_time > now:
        sleep_seconds = (target_time - now).total_seconds()
        time.sleep(sleep_seconds)


# === Financial Calculations ===

def calculate_pip_value(symbol: str, lot_size: float = 1.0, 
                       account_currency: str = "USD") -> float:
    """
    Calculate pip value for a currency pair.
    
    Args:
        symbol: Currency pair (e.g., "EURUSD")
        lot_size: Lot size
        account_currency: Account currency
    
    Returns:
        Pip value
    """
    # Simplified calculation - in practice, you'd need current exchange rates
    major_pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
    
    if symbol in major_pairs:
        if symbol.endswith("USD"):
            return 10 * lot_size  # For pairs where USD is quote currency
        elif symbol.startswith("USD"):
            return 10 * lot_size  # Simplified for USD base pairs
    
    return 10 * lot_size  # Default value


def calculate_position_size(account_balance: float, risk_percentage: float,
                          stop_loss_pips: float, pip_value: float) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        account_balance: Account balance
        risk_percentage: Risk percentage (0-100)
        stop_loss_pips: Stop loss in pips
        pip_value: Value per pip
    
    Returns:
        Position size in lots
    """
    risk_amount = account_balance * (risk_percentage / 100)
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    return round(position_size, 2)


def calculate_profit_loss(entry_price: float, exit_price: float,
                         position_size: float, pip_value: float,
                         position_type: str = "buy") -> float:
    """
    Calculate profit/loss for a trade.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        position_size: Position size in lots
        pip_value: Value per pip
        position_type: "buy" or "sell"
    
    Returns:
        Profit/loss amount
    """
    pip_difference = (exit_price - entry_price) * 10000  # Assuming 4-decimal pairs
    
    if position_type.lower() == "sell":
        pip_difference = -pip_difference
    
    return pip_difference * pip_value * position_size


def calculate_margin_required(position_size: float, current_price: float,
                            leverage: int = 100) -> float:
    """
    Calculate margin required for a position.
    
    Args:
        position_size: Position size in lots
        current_price: Current market price
        leverage: Leverage ratio
    
    Returns:
        Required margin
    """
    notional_value = position_size * 100000 * current_price  # 100000 = standard lot size
    return notional_value / leverage


# === Number and String Utilities ===

def round_to_decimals(value: float, decimals: int) -> float:
    """Round value to specified decimal places."""
    if decimals < 0:
        return value
    
    decimal_value = Decimal(str(value))
    rounded = decimal_value.quantize(
        Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP
    )
    return float(rounded)


def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """Format amount as currency string."""
    formatted = f"{amount:,.{decimals}f}"
    return f"{formatted} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def normalize_symbol(symbol: str) -> str:
    """Normalize trading symbol format."""
    # Remove common separators and convert to uppercase
    normalized = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    return normalized


# === File and Data Utilities ===

def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
              indent: int = 2, compress: bool = False) -> bool:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: File path
        indent: JSON indentation
        compress: Whether to compress the file
    
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = json.dumps(data, indent=indent, default=str)
        
        if compress:
            with gzip.open(f"{filepath}.gz", 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        logger.debug(f"Saved JSON data to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save JSON data: {e}")
        return False


def load_json(filepath: Union[str, Path], compressed: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        filepath: File path
        compressed: Whether file is compressed
    
    Returns:
        Loaded data or None if failed
    """
    try:
        filepath = Path(filepath)
        
        if compressed:
            with gzip.open(f"{filepath}.gz", 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        logger.debug(f"Loaded JSON data from {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        return None


def save_pickle(data: Any, filepath: Union[str, Path], compress: bool = True) -> bool:
    """Save data using pickle."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if compress:
            with gzip.open(f"{filepath}.pkl.gz", 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f)
        
        logger.debug(f"Saved pickle data to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save pickle data: {e}")
        return False


def load_pickle(filepath: Union[str, Path], compressed: bool = True) -> Optional[Any]:
    """Load data from pickle file."""
    try:
        filepath = Path(filepath)
        
        if compressed:
            with gzip.open(f"{filepath}.pkl.gz", 'rb') as f:
                data = pickle.load(f)
        else:
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
        
        logger.debug(f"Loaded pickle data from {filepath}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load pickle data: {e}")
        return None


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """Get hash of file contents."""
    hash_algo = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
    
    return hash_algo.hexdigest()


# === Decorators ===

def retry(max_attempts: int = 3, delay: float = 1.0, 
          exponential_backoff: bool = True, exceptions: Tuple = (Exception,)):
    """
    Retry decorator for functions.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Delay between attempts
        exponential_backoff: Whether to use exponential backoff
        exceptions: Exception types to catch
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


def timing(func: Callable):
    """Timing decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def rate_limit(calls_per_second: float):
    """Rate limiting decorator."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator


def cache_result(ttl_seconds: int = 300):
    """Simple caching decorator with TTL."""
    def decorator(func: Callable):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result is still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            return result
        
        return wrapper
    return decorator


# === Threading and Async Utilities ===

class ThreadSafeCounter:
    """Thread-safe counter."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value


@contextmanager
def thread_pool_executor(max_workers: int = None):
    """Context manager for ThreadPoolExecutor."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


def run_parallel(functions: List[Callable], max_workers: int = None, 
                timeout: Optional[float] = None) -> List[Any]:
    """
    Run functions in parallel using ThreadPoolExecutor.
    
    Args:
        functions: List of functions to run
        max_workers: Maximum number of worker threads
        timeout: Timeout for each function
    
    Returns:
        List of results
    """
    results = []
    
    with thread_pool_executor(max_workers) as executor:
        # Submit all functions
        future_to_func = {executor.submit(func): func for func in functions}
        
        # Collect results
        for future in as_completed(future_to_func, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Function failed: {e}")
                results.append(None)
    
    return results


# === Validation Utilities ===

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def validate_symbol_format(symbol: str) -> bool:
    """Validate trading symbol format."""
    # Basic validation for forex pairs
    symbol = symbol.upper()
    if len(symbol) == 6 and symbol.isalpha():
        return True
    return False


def validate_price(price: Union[int, float]) -> bool:
    """Validate price value."""
    return isinstance(price, (int, float)) and price > 0


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        Tuple of (is_valid, missing_keys)
    """
    missing_keys = [key for key in required_keys if key not in config]
    return len(missing_keys) == 0, missing_keys


# === Performance Monitoring ===

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'total': sum(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.start_times.clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


# === Utility Classes ===

class ConfigManager:
    """Simple configuration manager."""
    
    def __init__(self, config_file: Union[str, Path]):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            return load_json(self.config_file) or {}
        return {}
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        return save_json(self.config, self.config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        for key, value in updates.items():
            self.set(key, value)


if __name__ == "__main__":
    # Example usage and tests
    logger = get_logger("helpers_test")
    
    # Test timestamp functions
    current_time = get_current_timestamp()
    logger.info(f"Current timestamp: {current_time}")
    
    # Test market status
    market_open = is_market_open()
    logger.info(f"Market is open: {market_open}")
    
    # Test financial calculations
    position_size = calculate_position_size(10000, 2, 50, 10)
    logger.info(f"Calculated position size: {position_size}")
    
    # Test performance monitor
    performance_monitor.start_timer("test_operation")
    time.sleep(0.1)
    duration = performance_monitor.end_timer("test_operation")
    logger.info(f"Test operation took: {duration:.4f} seconds")
    
    # Test retry decorator
    @retry(max_attempts=3, delay=0.1)
    def failing_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"
    
    try:
        result = failing_function()
        logger.info(f"Retry test result: {result}")
    except:
        logger.info("Retry test failed after all attempts")
    
    # Test config manager
    config_mgr = ConfigManager("test_config.json")
    config_mgr.set("trading.risk_percentage", 2.0)
    config_mgr.set("api.timeout", 30)
    logger.info(f"Config test: {config_mgr.get('trading.risk_percentage')}")
    
    print("Helpers test completed successfully!")