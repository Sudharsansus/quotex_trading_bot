"""
Strategy Manager for Quotex Trading Bot
Handles multiple trading strategies, strategy selection, and performance tracking
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: TimeFrame = TimeFrame.M5
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, timeframe: TimeFrame = TimeFrame.M5):
        self.name = name
        self.timeframe = timeframe
        self.enabled = True
        self.performance = StrategyPerformance()
        self.parameters = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        pass
    
    def calculate_performance(self, trades: List[Dict]) -> StrategyPerformance:
        """Calculate strategy performance metrics"""
        if not trades:
            return self.performance
        
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        losing_trades = total_trades - winning_trades
        
        total_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
        total_loss = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum([trade.get('profit', 0) for trade in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        self.performance = StrategyPerformance(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit=total_profit,
            total_loss=total_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            last_updated=datetime.now()
        )
        
        return self.performance


class MovingAverageStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__("Moving Average Crossover", TimeFrame.M5)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'min_confidence': 0.6
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate MA crossover signal"""
        if len(data) < self.parameters['slow_period']:
            return None
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(window=self.parameters['slow_period']).mean()
        
        # Get current and previous values
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        current_price = data['close'].iloc[-1]
        
        # Check for crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            # Bullish crossover
            confidence = min(abs(current_fast - current_slow) / current_slow, 1.0)
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe
                )
        
        elif prev_fast >= prev_slow and current_fast < current_slow:
            # Bearish crossover
            confidence = min(abs(current_fast - current_slow) / current_slow, 1.0)
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe
                )
        
        return None
    
    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, rsi_period: int = 14):
        super().__init__("RSI Strategy", TimeFrame.M5)
        self.parameters = {
            'rsi_period': rsi_period,
            'overbought_level': 70,
            'oversold_level': 30,
            'min_confidence': 0.5
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate RSI-based signal"""
        if len(data) < self.parameters['rsi_period'] + 1:
            return None
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
        
        current_rsi = rsi.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Generate signals
        if current_rsi <= self.parameters['oversold_level']:
            # Oversold condition - potential buy
            confidence = (self.parameters['oversold_level'] - current_rsi) / self.parameters['oversold_level']
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    metadata={'rsi': current_rsi}
                )
        
        elif current_rsi >= self.parameters['overbought_level']:
            # Overbought condition - potential sell
            confidence = (current_rsi - self.parameters['overbought_level']) / (100 - self.parameters['overbought_level'])
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    metadata={'rsi': current_rsi}
                )
        
        return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        super().__init__("Bollinger Bands", TimeFrame.M5)
        self.parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'min_confidence': 0.6
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate Bollinger Bands signal"""
        if len(data) < self.parameters['bb_period']:
            return None
        
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.parameters['bb_period']).mean()
        std = data['close'].rolling(window=self.parameters['bb_period']).std()
        
        upper_band = sma + (std * self.parameters['bb_std'])
        lower_band = sma - (std * self.parameters['bb_std'])
        
        current_price = data['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        # Generate signals
        if current_price <= current_lower:
            # Price touched lower band - potential buy
            confidence = (current_lower - current_price) / (current_sma - current_lower)
            confidence = min(confidence, 1.0)
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    metadata={
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'sma': current_sma
                    }
                )
        
        elif current_price >= current_upper:
            # Price touched upper band - potential sell
            confidence = (current_price - current_upper) / (current_upper - current_sma)
            confidence = min(confidence, 1.0)
            if confidence >= self.parameters['min_confidence']:
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    entry_price=current_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    metadata={
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'sma': current_sma
                    }
                )
        
        return None
    
    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value


class StrategyManager:
    """Main strategy manager class"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        self.strategy_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        self.trade_history: Dict[str, List[Dict]] = {}
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info("Strategy Manager initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default trading strategies"""
        # Add default strategies
        self.add_strategy(MovingAverageStrategy(fast_period=10, slow_period=20))
        self.add_strategy(RSIStrategy(rsi_period=14))
        self.add_strategy(BollingerBandsStrategy(bb_period=20, bb_std=2.0))
        
        # Set default weights
        self.strategy_weights = {
            "Moving Average Crossover": 0.4,
            "RSI Strategy": 0.3,
            "Bollinger Bands": 0.3
        }
        
        # Enable all strategies by default
        self.active_strategies = list(self.strategies.keys())
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a new trading strategy"""
        self.strategies[strategy.name] = strategy
        self.performance_history[strategy.name] = []
        self.trade_history[strategy.name] = []
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a trading strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            if strategy_name in self.active_strategies:
                self.active_strategies.remove(strategy_name)
            if strategy_name in self.strategy_weights:
                del self.strategy_weights[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a trading strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            if strategy_name not in self.active_strategies:
                self.active_strategies.append(strategy_name)
            logger.info(f"Enabled strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a trading strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            if strategy_name in self.active_strategies:
                self.active_strategies.remove(strategy_name)
            logger.info(f"Disabled strategy: {strategy_name}")
    
    def update_strategy_weights(self, weights: Dict[str, float]):
        """Update strategy weights"""
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.strategy_weights = {k: v / total_weight for k, v in weights.items()}
            logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy_name in self.active_strategies:
            strategy = self.strategies[strategy_name]
            if strategy.enabled:
                try:
                    signal = strategy.generate_signal(data)
                    if signal:
                        signals.append(signal)
                        logger.debug(f"Generated signal from {strategy_name}: {signal.signal_type}")
                except Exception as e:
                    logger.error(f"Error generating signal from {strategy_name}: {str(e)}")
        
        return signals
    
    def get_consensus_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Get consensus signal from all active strategies"""
        signals = self.generate_signals(data)
        
        if not signals:
            return None
        
        # Calculate weighted consensus
        buy_weight = 0.0
        sell_weight = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_name, 0.0)
            weighted_confidence = signal.confidence * weight
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                buy_weight += weighted_confidence
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                sell_weight += weighted_confidence
            
            total_confidence += weighted_confidence
        
        # Determine consensus
        if buy_weight > sell_weight and buy_weight > 0.5:
            return TradingSignal(
                signal_type=SignalType.BUY,
                confidence=buy_weight,
                entry_price=data['close'].iloc[-1],
                strategy_name="Consensus",
                timeframe=TimeFrame.M5
            )
        elif sell_weight > buy_weight and sell_weight > 0.5:
            return TradingSignal(
                signal_type=SignalType.SELL,
                confidence=sell_weight,
                entry_price=data['close'].iloc[-1],
                strategy_name="Consensus",
                timeframe=TimeFrame.M5
            )
        
        return None
    
    def update_trade_result(self, strategy_name: str, trade_result: Dict):
        """Update trade result for a strategy"""
        if strategy_name in self.trade_history:
            self.trade_history[strategy_name].append(trade_result)
            
            # Update performance
            if strategy_name in self.strategies:
                performance = self.strategies[strategy_name].calculate_performance(
                    self.trade_history[strategy_name]
                )
                self.performance_history[strategy_name].append(performance)
                
                logger.info(f"Updated performance for {strategy_name}: "
                           f"Win Rate: {performance.win_rate:.2%}, "
                           f"Profit Factor: {performance.profit_factor:.2f}")
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].performance
        return None
    
    def get_all_performances(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies"""
        performances = {}
        for name, strategy in self.strategies.items():
            performances[name] = strategy.performance
        return performances
    
    def optimize_weights(self, lookback_days: int = 30):
        """Optimize strategy weights based on recent performance"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=lookback_days)
        
        strategy_scores = {}
        
        for strategy_name, strategy in self.strategies.items():
            # Get recent trades
            recent_trades = [
                trade for trade in self.trade_history.get(strategy_name, [])
                if trade.get('timestamp', current_time) >= cutoff_time
            ]
            
            if recent_trades:
                # Calculate performance score
                performance = strategy.calculate_performance(recent_trades)
                score = performance.win_rate * performance.profit_factor
                strategy_scores[strategy_name] = max(score, 0.1)  # Minimum weight
            else:
                strategy_scores[strategy_name] = 0.5  # Default weight
        
        # Normalize scores to weights
        if strategy_scores:
            self.update_strategy_weights(strategy_scores)
            logger.info("Optimized strategy weights based on recent performance")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of strategy manager"""
        return {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'strategy_weights': self.strategy_weights,
            'active_strategy_names': self.active_strategies,
            'performances': {name: {
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'total_trades': perf.total_trades
            } for name, perf in self.get_all_performances().items()}
        }