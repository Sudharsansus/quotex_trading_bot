"""
Utils package for the Quotex trading bot.
Provides logging, data processing, and general utility functions.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Team"

# Import main classes and functions for easy access
try:
    from .logger import (
        TradingBotLogger,
        setup_logging,
        get_logger,
        get_trading_logger,
        log_trade,
        log_signal,
        ContextLogger,
        log_execution_time
    )
except ImportError:
    # Fallback for direct execution
    try:
        from logger import (
            TradingBotLogger,
            setup_logging,
            get_logger,
            get_trading_logger,
            log_trade,
            log_signal,
            ContextLogger,
            log_execution_time
        )
    except ImportError as e:
        print(f"Warning: Could not import logger module: {e}")
        # Create dummy functions to prevent crashes
        def setup_logging(*args, **kwargs): pass
        def get_logger(*args, **kwargs): 
            import logging
            return logging.getLogger("dummy")
        def get_trading_logger(*args, **kwargs): 
            import logging
            return logging.getLogger("dummy_trading")
        def log_trade(*args, **kwargs): pass
        def log_signal(*args, **kwargs): pass

try:
    from .data_processor import (
        DataProcessor,
        detect_market_regime,
        calculate_fractal_dimension,
        validate_data_quality
    )
except ImportError:
    # Fallback for direct execution
    try:
        from data_processor import (
            DataProcessor,
            detect_market_regime,
            calculate_fractal_dimension,
            validate_data_quality
        )
    except ImportError as e:
        print(f"Warning: Could not import data_processor module: {e}")
        # Create dummy class
        class DataProcessor:
            def __init__(self, *args, **kwargs): pass
        def detect_market_regime(*args, **kwargs): pass
        def calculate_fractal_dimension(*args, **kwargs): return 0.5
        def validate_data_quality(*args, **kwargs): return {}

try:
    from .helpers import (
        # Time and date utilities
        get_current_timestamp,
        timestamp_to_datetime,
        datetime_to_timestamp,
        is_market_open,
        get_next_market_open,
        sleep_until,
        
        # Financial calculations
        calculate_pip_value,
        calculate_position_size,
        calculate_profit_loss,
        calculate_margin_required,
        
        # Number and string utilities
        round_to_decimals,
        format_currency,
        format_percentage,
        safe_divide,
        clamp,
        normalize_symbol,
        
        # File and data utilities
        save_json,
        load_json,
        save_pickle,
        load_pickle,
        ensure_directory,
        get_file_hash,
        
        # Decorators
        retry,
        timing,
        rate_limit,
        cache_result,
        
        # Threading utilities
        ThreadSafeCounter,
        thread_pool_executor,
        run_parallel,
        
        # Validation utilities
        validate_email,
        validate_url,
        validate_symbol_format,
        validate_price,
        validate_config,
        
        # Utility classes
        PerformanceMonitor,
        ConfigManager,
        performance_monitor
    )
except ImportError:
    # Fallback for direct execution
    try:
        from helpers import (
            get_current_timestamp,
            timestamp_to_datetime,
            datetime_to_timestamp,
            is_market_open,
            get_next_market_open,
            sleep_until,
            calculate_pip_value,
            calculate_position_size,
            calculate_profit_loss,
            calculate_margin_required,
            round_to_decimals,
            format_currency,
            format_percentage,
            safe_divide,
            clamp,
            normalize_symbol,
            save_json,
            load_json,
            save_pickle,
            load_pickle,
            ensure_directory,
            get_file_hash,
            retry,
            timing,
            rate_limit,
            cache_result,
            ThreadSafeCounter,
            thread_pool_executor,
            run_parallel,
            validate_email,
            validate_url,
            validate_symbol_format,
            validate_price,
            validate_config,
            PerformanceMonitor,
            ConfigManager,
            performance_monitor
        )
    except ImportError as e:
        print(f"Warning: Could not import helpers module: {e}")
        # Create dummy functions
        def get_current_timestamp(): 
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).isoformat()
        def is_market_open(*args, **kwargs): return True
        def calculate_position_size(*args, **kwargs): return 1.0
        def safe_divide(a, b, default=0): return a/b if b != 0 else default
        def retry(*args, **kwargs): 
            def decorator(func): return func
            return decorator
        def timing(func): return func
        class ConfigManager:
            def __init__(self, *args): pass
            def get(self, *args, **kwargs): return None
            def set(self, *args, **kwargs): pass

# Package-level exports - only include what was successfully imported
__all__ = []

# Add successfully imported items to __all__
for item_name in [
    # Logger exports
    "TradingBotLogger", "setup_logging", "get_logger", "get_trading_logger",
    "log_trade", "log_signal", "ContextLogger", "log_execution_time",
    
    # Data processor exports
    "DataProcessor", "detect_market_regime", "calculate_fractal_dimension", 
    "validate_data_quality",
    
    # Helper function exports
    "get_current_timestamp", "timestamp_to_datetime", "datetime_to_timestamp",
    "is_market_open", "get_next_market_open", "sleep_until",
    "calculate_pip_value", "calculate_position_size", "calculate_profit_loss",
    "calculate_margin_required", "round_to_decimals", "format_currency",
    "format_percentage", "safe_divide", "clamp", "normalize_symbol",
    "save_json", "load_json", "save_pickle", "load_pickle",
    "ensure_directory", "get_file_hash", "retry", "timing",
    "rate_limit", "cache_result", "ThreadSafeCounter", "thread_pool_executor",
    "run_parallel", "validate_email", "validate_url", "validate_symbol_format",
    "validate_price", "validate_config", "PerformanceMonitor", "ConfigManager",
    "performance_monitor"
]:
    if item_name in globals():
        __all__.append(item_name)

# Package initialization
def initialize_utils(log_level: str = "INFO", log_dir: str = "data/logs"):
    """
    Initialize the utils package with logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    
    Returns:
        TradingBotLogger instance or None if setup fails
    """
    try:
        return setup_logging(log_level, log_dir)
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        return None


def get_package_info() -> dict:
    """Get package information."""
    return {
        "name": "quotex_trading_bot.utils",
        "version": __version__,
        "author": __author__,
        "modules": ["logger", "data_processor", "helpers"],
        "exports_count": len(__all__)
    }


# Convenience functions for common operations
def quick_setup(log_level: str = "INFO", config_file: str = "config/settings.json") -> tuple:
    """
    Quick setup for common utils functionality.
    
    Args:
        log_level: Logging level
        config_file: Configuration file path
    
    Returns:
        Tuple of (logger, config_manager, data_processor)
    """
    try:
        # Setup logging
        logger_instance = setup_logging(log_level)
        
        # Setup configuration manager
        config_manager = ConfigManager(config_file)
        
        # Setup data processor
        data_processor = DataProcessor()
        
        # Get main logger
        logger = get_logger("main")
        logger.info("Utils package initialized successfully")
        
        return logger, config_manager, data_processor
    except Exception as e:
        print(f"Failed to setup utils: {e}")
        import logging
        return logging.getLogger("fallback"), None, None


# Common configurations
DEFAULT_LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_rotation": {
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
}

DEFAULT_DATA_CONFIG = {
    "scaler_type": "standard",
    "missing_data_method": "forward_fill",
    "outlier_threshold": 5.0,
    "validation_enabled": True
}

DEFAULT_TRADING_CONFIG = {
    "risk_percentage": 2.0,
    "max_positions": 5,
    "leverage": 100,
    "pip_value": 10.0,
    "minimum_lot_size": 0.01
}

# Export configurations
CONFIGS = {
    "logging": DEFAULT_LOG_CONFIG,
    "data": DEFAULT_DATA_CONFIG,
    "trading": DEFAULT_TRADING_CONFIG
}


def print_package_info():
    """Print package information."""
    print(f"\nQuotex Trading Bot Utils v{__version__}")
    print(f"Author: {__author__}")
    print(f"Successfully imported: {len(__all__)} items")
    print(f"Available modules: logger, data_processor, helpers")


# Simple test function
def test_imports():
    """Test if all imports are working."""
    print("Testing utils package imports...")
    
    # Test basic functions
    try:
        timestamp = get_current_timestamp()
        print(f"✓ Timestamp function works: {timestamp}")
    except Exception as e:
        print(f"✗ Timestamp function failed: {e}")
    
    try:
        market_status = is_market_open()
        print(f"✓ Market status function works: {market_status}")
    except Exception as e:
        print(f"✗ Market status function failed: {e}")
    
    try:
        logger = get_logger("test")
        print("✓ Logger function works")
    except Exception as e:
        print(f"✗ Logger function failed: {e}")
    
    print(f"Total exported items: {len(__all__)}")


if __name__ == "__main__":
    print_package_info()
    test_imports()