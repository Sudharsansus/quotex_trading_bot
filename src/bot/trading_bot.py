"""
Advanced Quotex Trading Bot
Core trading bot implementation with AI-powered decision making
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from ..api.quotex_api import QuotexAPI
from ..api.market_data import MarketDataManager
from ..indicators.signal_generator import SignalGenerator
from ..ml.prediction_engine import PredictionEngine
from .strategy_manager import StrategyManager
from .risk_manager import RiskManager
from ..utils.logger import setup_logger
from ..utils.data_processor import DataProcessor
from ..database.database_manager import DatabaseManager
from config.settings import TRADING_CONFIG, RISK_CONFIG, ML_CONFIG

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    asset: str
    direction: str  # 'CALL' or 'PUT'
    confidence: float
    entry_price: float
    expiry_time: int
    timestamp: datetime
    indicators: Dict[str, Any]
    ml_prediction: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0
    
class TradingBot:
    """Advanced AI-powered trading bot for Quotex"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.logger = setup_logger("TradingBot")
        self.db_manager = db_manager
        
        # Initialize components
        self.api = QuotexAPI()
        self.market_data = MarketDataManager()
        self.signal_generator = SignalGenerator()
        self.prediction_engine = PredictionEngine()
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.data_processor = DataProcessor()
        
        # Bot state
        self.is_running = False
        self.is_trading = False
        self.active_trades: Dict[str, Dict] = {}
        self.daily_stats = {
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'win_rate': 0.0
        }
        
        # Performance tracking
        self.account_balance = 0.0
        self.daily_start_balance = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        
        # Trade history
        self.trade_history: List[Dict] = []
        
    async def initialize(self):
        """Initialize the trading bot"""
        try:
            self.logger.info("🔧 Initializing trading bot components...")
            
            # Initialize API connection
            await self.api.connect()
            
            # Initialize market data streams
            await self.market_data.initialize(TRADING_CONFIG["assets"])
            
            # Initialize ML prediction engine
            await self.prediction_engine.initialize()
            
            # Get account information
            account_info = await self.api.get_account_info()
            self.account_balance = account_info.get('balance', 0)
            self.daily_start_balance = self.account_balance
            self.peak_balance = self.account_balance
            
            self.logger.info(f"💰 Account Balance: ${self.account_balance:.2f}")
            self.logger.info("✅ Trading bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Bot initialization failed: {e}")
            raise
    
    async def start_trading(self):
        """Start the main trading loop"""
        if self.is_running:
            self.logger.warning("⚠️ Trading bot is already running")
            return
        
        self.is_running = True
        self.is_trading = True
        
        self.logger.info("🚀 Starting trading operations...")
        
        try:
            # Start market data streams
            await self.market_data.start_streams()
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            self.is_running = False
            self.is_trading = False
    
    async def stop_trading(self):
        """Stop trading operations"""
        self.logger.info("🛑 Stopping trading operations...")
        self.is_trading = False
        
        # Close all active trades
        await self._close_all_trades()
        
        # Stop market data streams
        await self.market_data.stop_streams()
        
        # Disconnect API
        await self.api.disconnect()
        
        # Save final statistics
        await self._save_daily_stats()
        
        self.logger.info("✅ Trading stopped successfully")
    
    async def _main_trading_loop(self):
        """Main trading loop with signal generation and execution"""
        self.logger.info("🔄 Starting main trading loop...")
        
        while self.is_trading:
            try:
                # Check trading conditions
                if not await self._check_trading_conditions():
                    await asyncio.sleep(60)  # Wait 1 minute before next check
                    continue
                
                # Process each asset
                for asset in TRADING_CONFIG["assets"]:
                    if not self.is_trading:
                        break
                    
                    # Get latest market data
                    market_data = await self.market_data.get_latest_data(asset)
                    if market_data is None:
                        continue
                    
                    # Generate trading signal
                    signal = await self._generate_trading_signal(asset, market_data)
                    
                    if signal and signal.confidence >= TRADING_CONFIG["min_confidence"]:
                        # Execute trade
                        await self._execute_trade(signal)
                
                # Update bot statistics
                await self._update_statistics()
                
                # Check for trade exits
                await self._check_trade_exits()
                
                # Sleep before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(60)
    
    async def _generate_trading_signal(self, asset: str, market_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal using technical analysis and ML predictions"""
        try:
            # Generate technical indicators
            indicators = await self.signal_generator.generate_indicators(market_data)
            
            # Get ML prediction
            ml_prediction = await self.prediction_engine.predict(asset, market_data)
            
            # Combine signals using strategy manager
            combined_signal = await self.strategy_manager.combine_signals(
                asset, indicators, ml_prediction
            )
            
            if combined_signal is None:
                return None
            
            # Calculate risk score
            risk_score = await self.risk_manager.calculate_risk_score(
                asset, combined_signal, market_data
            )
            
            # Create trade signal
            signal = TradeSignal(
                asset=asset,
                direction=combined_signal['direction'],
                confidence=combined_signal['confidence'],
                entry_price=market_data.iloc[-1]['close'],
                expiry_time=combined_signal['expiry_time'],
                timestamp=datetime.now(),
                indicators=indicators,
                ml_prediction=ml_prediction,
                risk_score=risk_score
            )
            
            self.logger.info(f"📊 Generated signal for {asset}: {signal.direction} "
                           f"(Confidence: {signal.confidence:.2f}, Risk: {risk_score:.2f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {asset}: {e}")
            return None
    
    async def _execute_trade(self, signal: TradeSignal):
        """Execute a trade based on the signal"""
        try:
            # Check if we can open a new trade
            if not await self._can_open_trade(signal):
                return
            
            # Calculate trade amount
            trade_amount = await self.risk_manager.calculate_trade_amount(
                self.account_balance, signal.risk_score
            )
            
            # Place the trade
            trade_result = await self.api.place_trade(
                asset=signal.asset,
                amount=trade_amount,
                direction=signal.direction,
                expiry_time=signal.expiry_time
            )
            
            if trade_result and trade_result.get('success'):
                trade_id = trade_result.get('trade_id')
                
                # Store trade information
                trade_info = {
                    'id': trade_id,
                    'asset': signal.asset,
                    'direction': signal.direction,
                    'amount': trade_amount,
                    'entry_price': signal.entry_price,
                    'entry_time': signal.timestamp,
                    'expiry_time': signal.expiry_time,
                    'confidence': signal.confidence,
                    'risk_score': signal.risk_score,
                    'status': 'open'
                }
                
                self.active_trades[trade_id] = trade_info
                
                # Update statistics
                self.daily_stats['trades_count'] += 1
                
                # Save to database
                await self.db_manager.save_trade(trade_info)
                
                self.logger.info(f"✅ Trade executed: {signal.asset} {signal.direction} "
                               f"${trade_amount:.2f} (ID: {trade_id})")
                
            else:
                self.logger.warning(f"⚠️ Trade execution failed: {trade_result}")
                
        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
    
    async def _check_trade_exits(self):
        """Check for trade exits and update results"""
        completed_trades = []
        
        for trade_id, trade_info in self.active_trades.items():
            try:
                # Check if trade has expired
                if datetime.now() >= trade_info['expiry_time']:
                    # Get trade result
                    result = await self.api.get_trade_result(trade_id)
                    
                    if result:
                        # Update trade info
                        trade_info['exit_price'] = result.get('exit_price', 0)
                        trade_info['profit_loss'] = result.get('profit_loss', 0)
                        trade_info['status'] = result.get('status', 'closed')
                        trade_info['exit_time'] = datetime.now()
                        
                        # Update statistics
                        if trade_info['profit_loss'] > 0:
                            self.daily_stats['wins'] += 1
                        else:
                            self.daily_stats['losses'] += 1
                        
                        self.daily_stats['profit_loss'] += trade_info['profit_loss']
                        self.account_balance += trade_info['profit_loss']
                        
                        # Update win rate
                        total_trades = self.daily_stats['wins'] + self.daily_stats['losses']
                        if total_trades > 0:
                            self.daily_stats['win_rate'] = self.daily_stats['wins'] / total_trades
                        
                        # Add to trade history
                        self.trade_history.append(trade_info.copy())
                        
                        # Save to database
                        await self.db_manager.update_trade(trade_info)
                        
                        # Log result
                        result_emoji = "🟢" if trade_info['profit_loss'] > 0 else "🔴"
                        self.logger.info(f"{result_emoji} Trade closed: {trade_info['asset']} "
                                       f"P&L: ${trade_info['profit_loss']:.2f}")
                        
                        completed_trades.append(trade_id)
                        
            except Exception as e:
                self.logger.error(f"❌ Error checking trade {trade_id}: {e}")
        
        # Remove completed trades
        for trade_id in completed_trades:
            del self.active_trades[trade_id]
    
    async def _can_open_trade(self, signal: TradeSignal) -> bool:
        """Check if we can open a new trade"""
        # Check maximum concurrent trades
        if len(self.active_trades) >= TRADING_CONFIG["max_concurrent_trades"]:
            return False
        
        # Check maximum daily trades
        if self.daily_stats['trades_count'] >= TRADING_CONFIG["max_daily_trades"]:
            return False
        
        # Check daily loss limit
        if self.daily_stats['profit_loss'] <= -RISK_CONFIG["max_daily_loss"] * self.daily_start_balance:
            return False
        
        # Check maximum drawdown
        current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
        if current_drawdown > RISK_CONFIG["max_drawdown"]:
            return False
        
        # Check minimum confidence
        if signal.confidence < TRADING_CONFIG["min_confidence"]:
            return False
        
        # Check risk score
        if signal.risk_score > 0.8:  # High risk threshold
            return False
        
        return True
    
    async def _check_trading_conditions(self) -> bool:
        """Check if trading conditions are met"""
        # Check if market is open
        if not await self._is_market_open():
            return False
        
        # Check if we have sufficient balance
        if self.account_balance < TRADING_CONFIG["trade_amount"]:
            self.logger.warning("⚠️ Insufficient balance for trading")
            return False
        
        # Check if profit target is reached
        daily_profit = self.account_balance - self.daily_start_balance
        profit_target = RISK_CONFIG["profit_target"] * self.daily_start_balance
        
        if daily_profit >= profit_target:
            self.logger.info(f"🎯 Daily profit target reached: ${daily_profit:.2f}")
            return False
        
        return True
    
    async def _is_market_open(self) -> bool:
        """Check if market is open for trading"""
        current_time = datetime.now().time()
        start_time = datetime.strptime(TRADING_CONFIG["trading_hours"]["start"], "%H:%M").time()
        end_time = datetime.strptime(TRADING_CONFIG["trading_hours"]["end"], "%H:%M").time()
        
        return start_time <= current_time <= end_time
    
    async def _update_statistics(self):
        """Update bot statistics and performance metrics"""
        # Update peak balance
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        
        # Calculate current drawdown
        current_drawdown = (self.peak_balance - self.account_balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Log periodic statistics
        if self.daily_stats['trades_count'] > 0 and self.daily_stats['trades_count'] % 10 == 0:
            await self._log_statistics()
    
    async def _log_statistics(self):
        """Log current trading statistics"""
        daily_profit = self.account_balance - self.daily_start_balance
        daily_return = (daily_profit / self.daily_start_balance) * 100
        
        self.logger.info(f"""
📊 Trading Statistics:
   Balance: ${self.account_balance:.2f}
   Daily P&L: ${daily_profit:.2f} ({daily_return:.2f}%)
   Trades: {self.daily_stats['trades_count']}
   Win Rate: {self.daily_stats['win_rate']:.1%}
   Max Drawdown: {self.max_drawdown:.1%}
   Active Trades: {len(self.active_trades)}
        """)
    
    async def _save_daily_stats(self):
        """Save daily statistics to database"""
        try:
            daily_stats = {
                'date': datetime.now().date(),
                'starting_balance': self.daily_start_balance,
                'ending_balance': self.account_balance,
                'trades_count': self.daily_stats['trades_count'],
                'wins': self.daily_stats['wins'],
                'losses': self.daily_stats['losses'],
                'profit_loss': self.daily_stats['profit_loss'],
                'win_rate': self.daily_stats['win_rate'],
                'max_drawdown': self.max_drawdown,
                'peak_balance': self.peak_balance
            }
            
            await self.db_manager.save_daily_stats(daily_stats)
            self.logger.info("💾 Daily statistics saved")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving daily stats: {e}")
    
    async def _close_all_trades(self):
        """Close all active trades"""
        if not self.active_trades:
            return
        
        self.logger.info(f"🔄 Closing {len(self.active_trades)} active trades...")
        
        for trade_id in list(self.active_trades.keys()):
            try:
                await self.api.close_trade(trade_id)
                self.logger.info(f"✅ Trade {trade_id} closed")
            except Exception as e:
                self.logger.error(f"❌ Error closing trade {trade_id}: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        daily_profit = self.account_balance - self.daily_start_balance
        daily_return = (daily_profit / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        
        return {
            'account_balance': self.account_balance,
            'daily_profit': daily_profit,
            'daily_return': daily_return,
            'trades_count': self.daily_stats['trades_count'],
            'win_rate': self.daily_stats['win_rate'],
            'max_drawdown': self.max_drawdown,
            'active_trades': len(self.active_trades),
            'peak_balance': self.peak_balance
        }
    
    async def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        return self.trade_history[-limit:] if self.trade_history else []
    
    async def emergency_stop(self):
        """Emergency stop all trading operations"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        self.is_trading = False
        await self._close_all_trades()
        await self.stop_trading()
        self.logger.warning("🚨 Emergency stop completed")