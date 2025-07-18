"""
Advanced Quotex Trading Bot
Main entry point for the trading bot system
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.bot.trading_bot import TradingBot
from src.utils.logger import setup_logger
from config.settings import TRADING_CONFIG, ML_CONFIG
from src.database.database_manager import DatabaseManager

class QuotexTradingSystem:
    def __init__(self):
        self.logger = setup_logger("QuotexTradingSystem")
        self.bot = None
        self.db_manager = None
        self.running = False
        
    async def initialize(self):
        """Initialize all system components"""
        try:
            self.logger.info("🚀 Initializing Quotex Trading System...")
            
            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize trading bot
            self.bot = TradingBot(self.db_manager)
            await self.bot.initialize()
            
            self.logger.info("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def start_trading(self):
        """Start the trading system"""
        if not await self.initialize():
            return
        
        self.running = True
        self.logger.info("🎯 Starting trading operations...")
        
        try:
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start the main trading loop
            await self.bot.start_trading()
            
        except Exception as e:
            self.logger.error(f"❌ Trading error: {e}")
        finally:
            await self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
        
    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("🔄 Shutting down trading system...")
        
        if self.bot:
            await self.bot.stop_trading()
        
        if self.db_manager:
            await self.db_manager.close()
        
        self.logger.info("✅ System shutdown complete")

def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    QUOTEX TRADING BOT                         ║
    ║                   Advanced AI-Powered System                  ║
    ║                                                               ║
    ║  Features:                                                    ║
    ║  • Advanced Technical Analysis                                ║
    ║  • Machine Learning Predictions                               ║
    ║  • Risk Management System                                     ║
    ║  • Real-time Market Data                                      ║
    ║  • Multi-timeframe Analysis                                   ║
    ║                                                               ║
    ║  ⚠️  WARNING: Trading involves significant risk               ║
    ║      Only use with funds you can afford to lose              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create and run the trading system
    trading_system = QuotexTradingSystem()
    
    try:
        # Run the async event loop
        asyncio.run(trading_system.start_trading())
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()