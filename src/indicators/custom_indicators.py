"""
Custom Technical Indicators for Quotex Trading Bot
This module contains custom technical indicators for advanced trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional
import talib
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


class CustomIndicators:
    """
    Collection of custom technical indicators for trading analysis.
    """
    
    @staticmethod
    def heikin_ashi(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin Ashi candles for smoother price action analysis.
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with Heikin Ashi OHLC values
        """
        ha_data = pd.DataFrame(index=data.index)
        
        # Calculate Heikin Ashi values
        ha_data['ha_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_data['ha_open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
        ha_data['ha_open'].iloc[0] = data['open'].iloc[0]
        
        # Recalculate open based on previous values
        for i in range(1, len(ha_data)):
            ha_data['ha_open'].iloc[i] = (ha_data['ha_open'].iloc[i-1] + ha_data['ha_close'].iloc[i-1]) / 2
        
        ha_data['ha_high'] = np.maximum(data['high'], np.maximum(ha_data['ha_open'], ha_data['ha_close']))
        ha_data['ha_low'] = np.minimum(data['low'], np.minimum(ha_data['ha_open'], ha_data['ha_close']))
        
        return ha_data
    
    @staticmethod
    def supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator for trend direction and support/resistance.
        
        Args:
            data: DataFrame with OHLC columns
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            DataFrame with SuperTrend values and signals
        """
        # Calculate ATR
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
        
        # Calculate basic bands
        hl2 = (data['high'] + data['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        # Calculate SuperTrend
        for i in range(1, len(data)):
            if data['close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1  # Bullish
            elif data['close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1  # Bearish
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        # Set initial values
        direction.iloc[0] = 1
        supertrend.iloc[0] = lower_band.iloc[0]
        
        result = pd.DataFrame({
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
        
        return result
    
    @staticmethod
    def ichimoku_cloud(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with Ichimoku components
        """
        # Tenkan-sen (Conversion Line): 9-period high-low average
        tenkan_sen = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): 26-period high-low average
        kijun_sen = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period high-low average, shifted 26 periods forward
        senkou_span_b = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted 26 periods backward
        chikou_span = data['close'].shift(-26)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
    
    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: DataFrame with OHLC and volume columns
            
        Returns:
            Series with VWAP values
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    @staticmethod
    def fibonacci_retracement(data: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            data: DataFrame with OHLC columns
            period: Period to calculate high/low
            
        Returns:
            DataFrame with Fibonacci levels
        """
        high = data['high'].rolling(window=period).max()
        low = data['low'].rolling(window=period).min()
        
        diff = high - low
        
        fib_levels = pd.DataFrame({
            'fib_0': high,
            'fib_236': high - (diff * 0.236),
            'fib_382': high - (diff * 0.382),
            'fib_50': high - (diff * 0.5),
            'fib_618': high - (diff * 0.618),
            'fib_786': high - (diff * 0.786),
            'fib_100': low
        })
        
        return fib_levels
    
    @staticmethod
    def pivot_points(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pivot points and support/resistance levels.
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            DataFrame with pivot points
        """
        # Calculate pivot point
        pivot = (data['high'].shift(1) + data['low'].shift(1) + data['close'].shift(1)) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - data['low'].shift(1)
        s1 = 2 * pivot - data['high'].shift(1)
        r2 = pivot + (data['high'].shift(1) - data['low'].shift(1))
        s2 = pivot - (data['high'].shift(1) - data['low'].shift(1))
        r3 = data['high'].shift(1) + 2 * (pivot - data['low'].shift(1))
        s3 = data['low'].shift(1) - 2 * (data['high'].shift(1) - pivot)
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        })
    
    @staticmethod
    def money_flow_index(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            data: DataFrame with OHLC and volume columns
            period: Period for calculation
            
        Returns:
            Series with MFI values
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        # Calculate positive and negative money flow
        positive_mf = pd.Series(index=data.index, dtype=float)
        negative_mf = pd.Series(index=data.index, dtype=float)
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.iloc[i] = money_flow.iloc[i]
                negative_mf.iloc[i] = 0
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_mf.iloc[i] = 0
                negative_mf.iloc[i] = money_flow.iloc[i]
            else:
                positive_mf.iloc[i] = 0
                negative_mf.iloc[i] = 0
        
        # Calculate MFI
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        money_ratio = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def accumulation_distribution(data: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.
        
        Args:
            data: DataFrame with OHLC and volume columns
            
        Returns:
            Series with A/D values
        """
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)  # Handle division by zero
        
        ad = (clv * data['volume']).cumsum()
        return ad
    
    @staticmethod
    def chaikin_oscillator(data: pd.DataFrame, fast_period: int = 3, slow_period: int = 10) -> pd.Series:
        """
        Calculate Chaikin Oscillator.
        
        Args:
            data: DataFrame with OHLC and volume columns
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Series with Chaikin Oscillator values
        """
        ad = CustomIndicators.accumulation_distribution(data)
        
        fast_ema = ad.ewm(span=fast_period).mean()
        slow_ema = ad.ewm(span=slow_period).mean()
        
        return fast_ema - slow_ema
    
    @staticmethod
    def support_resistance_levels(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Identify support and resistance levels using local extrema.
        
        Args:
            data: DataFrame with OHLC columns
            window: Window for finding extrema
            
        Returns:
            DataFrame with support and resistance levels
        """
        # Find local maxima and minima
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Find peaks (resistance) and valleys (support)
        resistance_indices = argrelextrema(high_prices, np.greater, order=window)[0]
        support_indices = argrelextrema(low_prices, np.less, order=window)[0]
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['resistance'] = np.nan
        result['support'] = np.nan
        
        # Set resistance and support levels
        for idx in resistance_indices:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('resistance')] = high_prices[idx]
        
        for idx in support_indices:
            if idx < len(result):
                result.iloc[idx, result.columns.get_loc('support')] = low_prices[idx]
        
        # Forward fill to maintain levels
        result['resistance'] = result['resistance'].fillna(method='ffill')
        result['support'] = result['support'].fillna(method='ffill')
        
        return result
    
    @staticmethod
    def trend_strength(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate trend strength indicator.
        
        Args:
            data: DataFrame with OHLC columns
            period: Period for calculation
            
        Returns:
            Series with trend strength values (-1 to 1)
        """
        # Calculate price momentum
        momentum = data['close'].pct_change(period)
        
        # Calculate volatility
        volatility = data['close'].rolling(window=period).std()
        
        # Calculate trend strength
        trend_strength = momentum / volatility
        
        # Normalize to -1 to 1 range
        trend_strength = np.tanh(trend_strength)
        
        return trend_strength
    
    @staticmethod
    def volume_profile(data: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        """
        Calculate volume profile for price levels.
        
        Args:
            data: DataFrame with OHLC and volume columns
            bins: Number of price bins
            
        Returns:
            DataFrame with volume profile
        """
        # Create price bins
        price_min = data['low'].min()
        price_max = data['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Calculate volume for each price level
        volume_profile = pd.DataFrame({
            'price_level': (price_bins[:-1] + price_bins[1:]) / 2,
            'volume': 0.0
        })
        
        # Distribute volume across price levels
        for i in range(len(data)):
            typical_price = (data['high'].iloc[i] + data['low'].iloc[i] + data['close'].iloc[i]) / 3
            volume = data['volume'].iloc[i]
            
            # Find appropriate bin
            bin_idx = np.digitize(typical_price, price_bins) - 1
            bin_idx = max(0, min(bin_idx, len(volume_profile) - 1))
            
            volume_profile.loc[bin_idx, 'volume'] += volume
        
        return volume_profile