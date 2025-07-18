"""
Quotex Trading Bot - Main Package

A comprehensive trading bot for Quotex platform with advanced features:
- Machine learning-based predictions
- Technical indicator analysis
- Risk management
- Real-time market data processing
- Automated trading strategies
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"
__description__ = "Advanced Trading Bot for Quotex Platform"

# Package metadata
PACKAGE_INFO = {
    "name": "quotex_trading_bot",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "modules": [
        "api",        # Market data and Quotex API integration
        "bot",        # Trading bot core functionality
        "database",   # Database management
        "indicators", # Technical indicators
        "ml",         # Machine learning components
        "utils"       # Utilities and helpers
    ]
}

# Try to import utils package
try:
    from . import utils
    utils_available = True
except ImportError as e:
    print(f"Warning: Utils package not available: {e}")
    utils_available = False

# Try to import other main modules
try:
    from . import api
    api_available = True
except ImportError:
    api_available = False

try:
    from . import bot
    bot_available = True
except ImportError:
    bot_available = False

try:
    from . import database
    database_available = True
except ImportError:
    database_available = False

try:
    from . import indicators
    indicators_available = True
except ImportError:
    indicators_available = False

try:
    from . import ml
    ml_available = True
except ImportError:
    ml_available = False

# Package status
PACKAGE_STATUS = {
    "utils": utils_available,
    "api": api_available, 
    "bot": bot_available,
    "database": database_available,
    "indicators": indicators_available,
    "ml": ml_available
}

def get_package_status():
    """Get the status of all package modules."""
    return PACKAGE_STATUS

def print_package_status():
    """Print the status of all package modules."""
    print(f"\n{PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Author: {PACKAGE_INFO['author']}")
    print("\nModule Status:")
    for module, available in PACKAGE_STATUS.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {module}: {status}")

def setup_project():
    """
    Initialize the trading bot project.
    This function can be called to set up logging, create directories, etc.
    """
    print("Setting up Quotex Trading Bot...")
    
    # Create necessary directories
    import os
    directories = [
        "data/logs",
        "data/historical", 
        "data/models",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Initialize logging if utils is available
    if utils_available:
        try:
            from .utils import setup_logging
            setup_logging("INFO", "data/logs")
            print("✓ Logging initialized")
        except Exception as e:
            print(f"✗ Failed to setup logging: {e}")
    
    print("Project setup completed!")

# Convenience imports - only if modules are available
if utils_available:
    try:
        from .utils import get_logger, setup_logging
    except ImportError:
        pass

# Main exports
__all__ = [
    "PACKAGE_INFO",
    "PACKAGE_STATUS", 
    "get_package_status",
    "print_package_status",
    "setup_project"
]

# Add available module imports to exports
if utils_available:
    __all__.extend(["utils"])
if api_available:
    __all__.extend(["api"])
if bot_available:
    __all__.extend(["bot"])
if database_available:
    __all__.extend(["database"])
if indicators_available:
    __all__.extend(["indicators"])
if ml_available:
    __all__.extend(["ml"])

if __name__ == "__main__":
    print_package_status()