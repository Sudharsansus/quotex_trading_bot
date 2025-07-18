"""
Signal Generator for Quotex Trading Bot
This module generates trading signals based on technical indicators and custom strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import talib
from .custom_indicators import CustomIndicators
from .technical_indicators import TechnicalIndicators
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading decisions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class TradingSignal:
    """Class to represent a trading signal."""
    
    def __init__(self, signal_type: SignalType, strength: SignalStrength, 
                 confidence: float, price: float, timestamp: pd.Timestamp,
                 indicators: Dict, reason: str = ""):
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence  # 0.0 to 1.0
        self.price = price
        self.timestamp = timestamp
        self.indicators = indicators
        self.reason = reason
    
    def __repr__(self):
        return f"TradingSignal({self.signal_type.value}, {self.strength.value}, {self.confidence:.2f})"


class SignalGenerator:
    """
    Main signal generator class that combines multiple indicators to generate trading signals.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or self._default_config()
        self.custom_indicators = CustomIndicators()
        self.technical_indicators = TechnicalIndicators()
        
    def _default_config(self) -> Dict:
        """Default configuration for signal generation."""
        return {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'adx_period': 14,
            'adx_threshold': 25,
            'volume_threshold': 1.5,
            'trend_strength_threshold': 0.3,
            'signal_cooldown': 5,  # minutes
            'min_confidence': 0.6
        }
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with OHLC and volume data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Calculate all indicators
        indicators = self._calculate_indicators(data)
        
        # Generate signals for each timestamp
        for i in range(len(data)):
            if i < 50:  # Skip first 50 candles for indicator warmup
                continue
                
            signal = self._generate_signal_for_timestamp(data, indicators, i)
            if signal and signal.confidence >= self.config['min_confidence']:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators."""
        indicators = {}
        
        # Basic indicators
        indicators['rsi'] = talib.RSI(data['close'], timeperiod=self.config['rsi_period'])
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(
            data['close'], 
            fastperiod=self.config['macd_fast'],
            slowperiod=self.config['macd_slow'],
            signalperiod=self.config['macd_signal']
        )
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(
            data['close'],
            timeperiod=self.config['bb_period'],
            nbdevup=self.config['bb_std'],
            nbdevdn=self.config['bb_std']
        )
        
        # Stochastic
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
            data['high'], data['low'], data['close'],
            fastk_period=self.config['stoch_k'],
            slowk_period=self.config['stoch_d'],
            slowd_period=self.config['stoch_d']
        )
        
        # ADX
        indicators['adx'] = talib.ADX(
            data['high'], data['low'], data['close'],
            timeperiod=self.config['adx_period']
        )
        
        # Moving Averages
        indicators['sma_20'] = talib.SMA(data['close'], timeperiod=20)
        indicators['sma_50'] = talib.SMA(data['close'], timeperiod=50)
        indicators['ema_12'] = talib.EMA(data['close'], timeperiod=12)
        indicators['ema_26'] = talib.EMA(data['close'], timeperiod=26)
        
        # Volume indicators
        indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
        
        # Custom indicators
        indicators['supertrend'] = self.custom_indicators.supertrend(data)
        indicators['ichimoku'] = self.custom_indicators.ichimoku_cloud(data)
        indicators['heikin_ashi'] = self.custom_indicators.heikin_ashi(data)
        indicators['mfi'] = self.custom_indicators.money_flow_index(data)
        indicators['trend_strength'] = self.custom_indicators.trend_strength(data)
        indicators['support_resistance'] = self.custom_indicators.support_resistance_levels(data)
        
        return indicators
    
    def _generate_signal_for_timestamp(self, data: pd.DataFrame, indicators: Dict, idx: int) -> Optional[TradingSignal]:
        """Generate signal for a specific timestamp."""
        
        # Get current values
        current_price = data['close'].iloc[idx]
        current_time = data.index[idx]
        
        # Calculate individual signal components
        rsi_signal = self._rsi_signal(indicators['rsi'].iloc[idx])
        macd_signal = self._macd_signal(
            indicators['macd'].iloc[idx],
            indicators['macd_signal'].iloc[idx],
            indicators['macd_hist'].iloc[idx]
        )
        bb_signal = self._bollinger_bands_signal(
            current_price,
            indicators['bb_upper'].iloc[idx],
            indicators['bb_middle'].iloc[idx],
            indicators['bb_lower'].iloc[idx]
        )
        stoch_signal = self._stochastic_signal(
            indicators['stoch_k'].iloc[idx],
            indicators['stoch_d'].iloc[idx]
        )
        trend_signal = self._trend_signal(
            indicators['sma_20'].iloc[idx],
            indicators['sma_50'].iloc[idx],
            current_price
        )
        volume_signal = self._volume_signal(indicators['volume_ratio'].iloc[idx])
        supertrend_signal = self._supertrend_signal(indicators['supertrend'], idx)
        
        # Combine signals
        signal_scores = {
            'rsi': rsi_signal,
            'macd': macd_signal,
            'bb': bb_signal,
            'stoch': stoch_signal,
            'trend': trend_signal,
            'volume': volume_signal,
            'supertrend': supertrend_signal
        }
        
        # Calculate overall signal
        overall_signal, confidence = self._combine_signals(signal_scores)
        
        # Determine signal strength
        strength = self._calculate_signal_strength(signal_scores, confidence)
        
        # Create reason string
        reason = self._create_reason_string(signal_scores)
        
        if overall_signal != SignalType.HOLD:
            return TradingSignal(
                signal_type=overall_signal,
                strength=strength,
                confidence=confidence,
                price=current_price,
                timestamp=current_time,
                indicators=signal_scores,
                reason=reason
            )
        
        return None
    
    def _rsi_signal(self, rsi_value: float) -> float:
        """Generate RSI signal (-1 to 1)."""
        if pd.isna(rsi_value):
            return 0.0
        
        if rsi_value <= self.config['rsi_oversold']:
            return 1.0  # Buy signal
        elif rsi_value >= self.config['rsi_overbought']:
            return -1.0  # Sell signal
        else:
            # Gradual transition
            if rsi_value < 50:
                return (50 - rsi_value) / (50 - self.config['rsi_oversold'])
            else:
                return -(rsi_value - 50) / (self.config['rsi_overbought'] - 50)
    
    def _macd_signal(self, macd: float, signal: float, histogram: float) -> float:
        """Generate MACD signal (-1 to 1)."""
        if pd.isna(macd) or pd.isna(signal) or pd.isna(histogram):
            return 0.0
        
        # MACD crossover
        if macd > signal and histogram > 0:
            return min(1.0, histogram * 10)  # Buy signal
        elif macd < signal and histogram < 0:
            return max(-1.0, histogram * 10)  # Sell signal
        else:
            return 0.0
    
    def _bollinger_bands_signal(self, price: float, upper: float, middle: float, lower: float) -> float:
        """Generate Bollinger Bands signal (-1 to 1)."""
        if pd.isna(upper) or pd.isna(lower) or pd.isna(middle):
            return 0.0
        
        band_width = upper - lower
        if band_width == 0:
            return 0.0
        
        # Calculate position within bands
        position = (price - lower) / band_width
        
        if position <= 0.1:  # Near lower band
            return 0.8  # Buy signal
        elif position >= 0.9:  # Near upper band
            return -0.8  # Sell signal
        else:
            return 0.0
    
    def _stochastic_signal(self, k: float, d: float) -> float:
        """Generate Stochastic signal (-1 to 1)."""
        if pd.isna(k) or pd.isna(d):
            return 0.0
        
        if k <= 20 and d <= 20:
            return 0.7  # Buy signal
        elif k >= 80 and d >= 80:
            return -0.7  # Sell signal
        else:
            return 0.0
    
    def _trend_signal(self, sma_20: float, sma_50: float, price: float) -> float:
        """Generate trend signal (-1 to 1)."""
        if pd.isna(sma_20) or pd.isna(sma_50):
            return 0.0
        
        # Moving average crossover
        if sma_20 > sma_50 and price > sma_20:
            return 0.6  # Bullish trend
        elif sma_20 < sma_50 and price < sma_20:
            return -0.6  # Bearish trend
        else:
            return 0.0
    
    def _volume_signal(self, volume_ratio: float) -> float:
        """Generate volume signal (-1 to 1)."""
        if pd.isna(volume_ratio):
            return 0.0
        
        if volume_ratio > self.config['volume_threshold']:
            return 0.3  # High volume confirmation
        else:
            return 0.0
    
    def _supertrend_signal(self, supertrend_data: pd.DataFrame, idx: int) -> float:
        """Generate SuperTrend signal (-1 to 1)."""
        if idx >= len(supertrend_data):
            return 0.0
        
        direction = supertrend_data['direction'].iloc[idx]
        if pd.isna(direction):
            return 0.0
        
        return 0.5 * direction  # 0.5 for buy, -0.5 for sell
    
    def _combine_signals(self, signal_scores: Dict[str, float]) -> Tuple[SignalType, float]:
        """Combine individual signals into overall signal."""
        
        # Weighted combination
        weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bb': 0.15,
            'stoch': 0.15,
            'trend': 0.15,
            'volume': 0.05,
            'supertrend': 0.1
        }
        
        weighted_score = sum(signal_scores[key] * weights[key] for key in weights.keys())
        
        # Calculate confidence based on signal agreement
        positive_signals = sum(1 for score in signal_scores.values() if score > 0.1)
        negative_signals = sum(1 for score in signal_scores.values() if score < -0.1)
        total_signals = len(signal_scores)
        
        if positive_signals > negative_signals:
            confidence = positive_signals / total_signals
            signal_type = SignalType.STRONG_BUY if weighted_score > 0.5 else SignalType.BUY
        elif negative_signals > positive_signals:
            confidence = negative_signals / total_signals
            signal_type = SignalType.STRONG_SELL if weighted_score < -0.5 else SignalType.SELL
        else:
            confidence = 0.0
            signal_type = SignalType.HOLD
        
        return signal_type, confidence
    
    def _calculate_signal_strength(self, signal_scores: Dict[str, float], confidence: float) -> SignalStrength:
        """Calculate signal strength based on scores and confidence."""
        
        avg_score = abs(np.mean(list(signal_scores.values())))
        
        if confidence >= 0.8 and avg_score >= 0.6:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.7 and avg_score >= 0.4:
            return SignalStrength.STRONG
        elif confidence >= 0.6 and avg_score >= 0.2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _create_reason_string(self, signal_scores: Dict[str, float]) -> str:
        """Create a human-readable reason string for the signal."""
        
        positive_indicators = []
        negative_indicators = []
        
        for indicator, score in signal_scores.items():
            if score > 0.1:
                positive_indicators.append(f"{indicator.upper()}({score:.2f})")
            elif score < -0.1:
                negative_indicators.append(f"{indicator.upper()}({score:.2f})")
        
        reason_parts = []
        if positive_indicators:
            reason_parts.append(f"Bullish: {', '.join(positive_indicators)}")
        if negative_indicators:
            reason_parts.append(f"Bearish: {', '.join(negative_indicators)}")
        
        return " | ".join(reason_parts) if reason_parts else "Neutral signals"
    
    def get_signal_summary(self, signals: List[TradingSignal]) -> Dict:
        """Get summary statistics of generated signals."""
        
        if not signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'strong_signals': 0,
                'avg_confidence': 0.0
            }
        
        buy_count = sum(1 for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY])
        sell_count = sum(1 for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL])
        strong_count = sum(1 for s in signals if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'strong_signals': strong_count,
            'avg_confidence': avg_confidence,
            'signal_distribution': {
                'STRONG_BUY': sum(1 for s in signals if s.signal_type == SignalType.STRONG_BUY),
                'BUY': sum(1 for s in signals if s.signal_type == SignalType.BUY),
                'SELL': sum(1 for s in signals if s.signal_type == SignalType.SELL),
                'STRONG_SELL': sum(1 for s in signals if s.signal_type == SignalType.STRONG_SELL)
            }
        }
    
    def filter_signals_by_timeframe(self, signals: List[TradingSignal], 
                                   timeframe: str = '5T') -> List[TradingSignal]:
        """Filter signals to avoid too frequent signals in the same timeframe."""
        
        if not signals:
            return signals
        
        filtered_signals = []
        last_signal_time = None
        
        for signal in signals:
            if last_signal_time is None:
                filtered_signals.append(signal)
                last_signal_time = signal.timestamp
            else:
                # Check if enough time has passed
                time_diff = signal.timestamp - last_signal_time
                if time_diff >= pd.Timedelta(timeframe):
                    filtered_signals.append(signal)
                    last_signal_time = signal.timestamp
        
        return filtered_signals
    
    def get_current_market_sentiment(self, data: pd.DataFrame) -> Dict:
        """Analyze current market sentiment based on recent data."""
        
        if len(data) < 20:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0}
        
        # Calculate indicators for sentiment analysis
        indicators = self._calculate_indicators(data)
        
        # Get recent values (last 5 candles)
        recent_idx = len(data) - 5
        
        sentiment_scores = []
        
        # RSI sentiment
        rsi_recent = indicators['rsi'].iloc[recent_idx:].mean()
        if rsi_recent < 30:
            sentiment_scores.append(1.0)  # Bullish (oversold)
        elif rsi_recent > 70:
            sentiment_scores.append(-1.0)  # Bearish (overbought)
        else:
            sentiment_scores.append(0.0)
        
        # MACD sentiment
        macd_recent = indicators['macd'].iloc[recent_idx:].mean()
        macd_signal_recent = indicators['macd_signal'].iloc[recent_idx:].mean()
        if macd_recent > macd_signal_recent:
            sentiment_scores.append(0.5)
        else:
            sentiment_scores.append(-0.5)
        
        # Trend sentiment
        sma_20_recent = indicators['sma_20'].iloc[recent_idx:].mean()
        sma_50_recent = indicators['sma_50'].iloc[recent_idx:].mean()
        current_price = data['close'].iloc[-1]
        
        if current_price > sma_20_recent > sma_50_recent:
            sentiment_scores.append(1.0)
        elif current_price < sma_20_recent < sma_50_recent:
            sentiment_scores.append(-1.0)
        else:
            sentiment_scores.append(0.0)
        
        # Volume sentiment
        volume_ratio_recent = indicators['volume_ratio'].iloc[recent_idx:].mean()
        if volume_ratio_recent > 1.5:
            sentiment_scores.append(0.3)  # High volume is generally bullish
        else:
            sentiment_scores.append(0.0)
        
        # Calculate overall sentiment
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_confidence = 1.0 - (np.std(sentiment_scores) / 2.0)  # Lower std = higher confidence
        
        if avg_sentiment > 0.2:
            sentiment = 'BULLISH'
        elif avg_sentiment < -0.2:
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'confidence': max(0.0, min(1.0, sentiment_confidence)),
            'score': avg_sentiment,
            'components': {
                'rsi': sentiment_scores[0],
                'macd': sentiment_scores[1],
                'trend': sentiment_scores[2],
                'volume': sentiment_scores[3]
            }
        }
    
    def backtest_signals(self, data: pd.DataFrame, signals: List[TradingSignal], 
                        initial_balance: float = 10000.0) -> Dict:
        """Simple backtest of generated signals."""
        
        balance = initial_balance
        position = 0  # 0 = no position, 1 = long, -1 = short
        trades = []
        entry_price = 0.0
        
        for signal in signals:
            current_price = signal.price
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] and position <= 0:
                # Enter long position
                if position == -1:  # Close short position
                    pnl = entry_price - current_price
                    balance += pnl
                    trades.append({
                        'type': 'SHORT_CLOSE',
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'timestamp': signal.timestamp
                    })
                
                # Enter long
                position = 1
                entry_price = current_price
                trades.append({
                    'type': 'LONG_OPEN',
                    'price': current_price,
                    'pnl': 0,
                    'balance': balance,
                    'timestamp': signal.timestamp
                })
                
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL] and position >= 0:
                # Enter short position
                if position == 1:  # Close long position
                    pnl = current_price - entry_price
                    balance += pnl
                    trades.append({
                        'type': 'LONG_CLOSE',
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'timestamp': signal.timestamp
                    })
                
                # Enter short
                position = -1
                entry_price = current_price
                trades.append({
                    'type': 'SHORT_OPEN',
                    'price': current_price,
                    'pnl': 0,
                    'balance': balance,
                    'timestamp': signal.timestamp
                })
        
        # Calculate performance metrics
        total_return = (balance - initial_balance) / initial_balance * 100
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in trades if trade['pnl'] < 0)
        total_trades = len([trade for trade in trades if trade['pnl'] != 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'trades': trades
        }
    
    def optimize_parameters(self, data: pd.DataFrame, parameter_ranges: Dict) -> Dict:
        """Optimize signal generation parameters using grid search."""
        
        best_params = None
        best_performance = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(parameter_ranges)
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"Testing combination {i+1}/{len(param_combinations)}")
            
            # Update configuration
            old_config = self.config.copy()
            self.config.update(params)
            
            try:
                # Generate signals with new parameters
                signals = self.generate_signals(data)
                
                # Backtest signals
                backtest_result = self.backtest_signals(data, signals)
                
                # Use total return as performance metric
                performance = backtest_result['total_return']
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Error testing parameters {params}: {e}")
                continue
            finally:
                # Restore original configuration
                self.config = old_config
        
        logger.info(f"Best parameters found with {best_performance:.2f}% return")
        
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'total_combinations_tested': len(param_combinations)
        }
    
    def _generate_param_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters for optimization."""
        
        import itertools
        
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def export_signals_to_csv(self, signals: List[TradingSignal], filename: str):
        """Export signals to CSV file."""
        
        if not signals:
            logger.warning("No signals to export")
            return
        
        data = []
        for signal in signals:
            data.append({
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.value,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'price': signal.price,
                'reason': signal.reason
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(signals)} signals to {filename}")
    
    def load_signals_from_csv(self, filename: str) -> List[TradingSignal]:
        """Load signals from CSV file."""
        
        try:
            df = pd.read_csv(filename)
            signals = []
            
            for _, row in df.iterrows():
                signal = TradingSignal(
                    signal_type=SignalType(row['signal_type']),
                    strength=SignalStrength(row['strength']),
                    confidence=row['confidence'],
                    price=row['price'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    indicators={},
                    reason=row['reason']
                )
                signals.append(signal)
            
            logger.info(f"Loaded {len(signals)} signals from {filename}")
            return signals
            
        except Exception as e:
            logger.error(f"Error loading signals from {filename}: {e}")
            return []