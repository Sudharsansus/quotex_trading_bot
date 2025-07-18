"""
Technical Indicators Module for Quotex Trading Bot
Provides standard technical analysis indicators using numpy and pandas
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    A comprehensive collection of technical indicators for trading analysis.
    All methods are static and can be used independently.
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: Price series (typically close prices)
            period: Period for moving average
            
        Returns:
            pd.Series: Simple moving average values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: Price series (typically close prices)
            period: Period for EMA calculation
            
        Returns:
            pd.Series: Exponential moving average values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Weighted Moving Average
        
        Args:
            data: Price series
            period: Period for WMA calculation
            
        Returns:
            pd.Series: Weighted moving average values
        """
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series (typically close prices)
            period: Period for RSI calculation
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD line, Signal line, Histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            data: Price series
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: Upper band, Middle band (SMA), Lower band
        """
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation
            d_period: Period for %D calculation
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K, %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for ATR calculation
            
        Returns:
            pd.Series: ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average Directional Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for ADX calculation
            
        Returns:
            pd.Series: ADX values
        """
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        # Calculate directional movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Only keep positive directional movements
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate directional indicators
        di_plus = 100 * (dm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum())
        di_minus = 100 * (dm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum())
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for CCI calculation
            
        Returns:
            pd.Series: CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: pd.Series(x).mad(), raw=False
        )
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for Williams %R calculation
            
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change
        
        Args:
            data: Price series
            period: Period for ROC calculation
            
        Returns:
            pd.Series: ROC values
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum
        
        Args:
            data: Price series
            period: Period for momentum calculation
            
        Returns:
            pd.Series: Momentum values
        """
        return data - data.shift(period)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            pd.Series: OBV values
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Money Flow Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Period for MFI calculation
            
        Returns:
            pd.Series: MFI values
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """
        Parabolic SAR
        
        Args:
            high: High price series
            low: Low price series
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
            
        Returns:
            pd.Series: Parabolic SAR values
        """
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        af = pd.Series(index=high.index, dtype=float)
        ep = pd.Series(index=high.index, dtype=float)
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        af.iloc[0] = acceleration
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(high)):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
                
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = low.iloc[i]
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] - af.iloc[i-1] * (sar.iloc[i-1] - ep.iloc[i-1])
                
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = high.iloc[i]
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        return sar
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> dict:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: High price
            low: Low price
            
        Returns:
            dict: Fibonacci levels
        """
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> dict:
        """
        Calculate pivot points and support/resistance levels
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            dict: Pivot points and levels
        """
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                      conversion_period: int = 9, base_period: int = 26,
                      leading_span_b_period: int = 52, displacement: int = 26) -> dict:
        """
        Ichimoku Cloud components
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            conversion_period: Conversion line period
            base_period: Base line period
            leading_span_b_period: Leading span B period
            displacement: Displacement for leading spans
            
        Returns:
            dict: Ichimoku components
        """
        # Conversion Line (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion_period).max() + 
                          low.rolling(window=conversion_period).min()) / 2
        
        # Base Line (Kijun-sen)
        base_line = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_period).max() + 
                          low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-displacement)
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        }