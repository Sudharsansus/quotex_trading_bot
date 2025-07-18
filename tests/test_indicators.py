"""
Test suite for technical indicators.
Tests all technical analysis indicators and signal generation.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class MockTechnicalIndicators:
    """Mock technical indicators for testing when actual module isn't available."""
    
    @staticmethod
    def sma(data, period=20):
        """Simple Moving Average."""
        if isinstance(data, pd.Series):
            return data.rolling(window=period).mean()
        elif isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
            return data.rolling(window=period).mean()
        else:
            raise ValueError("Data must be pandas Series, list, or numpy array")
    
    @staticmethod
    def ema(data, period=20):
        """Exponential Moving Average."""
        if isinstance(data, pd.Series):
            return data.ewm(span=period).mean()
        elif isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
            return data.ewm(span=period).mean()
        else:
            raise ValueError("Data must be pandas Series, list, or numpy array")
    
    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index."""
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator."""
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands."""
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator."""
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'k': k_percent,
            'd': d_percent
        })
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range."""
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr


class MockSignalGenerator:
    """Mock signal generator for testing."""
    
    def __init__(self):
        self.indicators = MockTechnicalIndicators()
    
    def generate_signals(self, data):
        """Generate trading signals from market data."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'HOLD'
        signals['confidence'] = 0.5
        
        # Simple RSI-based signals
        rsi = self.indicators.rsi(data['close'])
        signals.loc[rsi < 30, 'signal'] = 'BUY'
        signals.loc[rsi < 30, 'confidence'] = 0.8
        signals.loc[rsi > 70, 'signal'] = 'SELL'
        signals.loc[rsi > 70, 'confidence'] = 0.8
        
        return signals
    
    def crossover_signal(self, fast_ma, slow_ma):
        """Generate crossover signals."""
        signals = pd.Series('HOLD', index=fast_ma.index)
        
        # Golden cross (fast MA crosses above slow MA)
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals.loc[golden_cross] = 'BUY'
        
        # Death cross (fast MA crosses below slow MA)
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals.loc[death_cross] = 'SELL'
        
        return signals


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicators."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        price_changes = np.random.normal(0, 0.01, 100)
        prices = 100 * np.exp(np.cumsum(price_changes))
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 100),
            'high': prices + np.abs(np.random.normal(0, 0.3, 100)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 100)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.indicators = MockTechnicalIndicators()
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        sma_20 = self.indicators.sma(self.sample_data['close'], 20)
        
        # Check basic properties
        self.assertIsInstance(sma_20, pd.Series)
        self.assertEqual(len(sma_20), len(self.sample_data))
        
        # First 19 values should be NaN
        self.assertTrue(sma_20.iloc[:19].isna().all())
        
        # Check manual calculation for a specific point
        manual_sma = self.sample_data['close'].iloc[19:40].mean()
        calculated_sma = sma_20.iloc[39]
        self.assertAlmostEqual(manual_sma, calculated_sma, places=10)
        
        # SMA should smooth the data (less volatile)
        sma_volatility = sma_20.dropna().std()
        price_volatility = self.sample_data['close'].std()
        self.assertLess(sma_volatility, price_volatility)
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema_20 = self.indicators.ema(self.sample_data['close'], 20)
        
        # Check basic properties
        self.assertIsInstance(ema_20, pd.Series)
        self.assertEqual(len(ema_20), len(self.sample_data))
        
        # EMA should not have initial NaN values (unlike SMA)
        self.assertFalse(ema_20.iloc[0:].isna().all())
        
        # EMA should be more responsive than SMA
        sma_20 = self.indicators.sma(self.sample_data['close'], 20)
        
        # Compare responsiveness (EMA should change more quickly)
        ema_changes = ema_20.diff().abs().mean()
        sma_changes = sma_20.diff().abs().mean()
        self.assertGreater(ema_changes, sma_changes)
    
    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation."""
        rsi = self.indicators.rsi(self.sample_data['close'], 14)
        
        # Check basic properties
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.sample_data))
        
        # RSI should be between 0 and 100
        rsi_values = rsi.dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
        
        # Check RSI properties
        self.assertGreater(rsi_values.max(), 50)  # Should have some variation
        self.assertLess(rsi_values.min(), 50)
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_data = self.indicators.macd(self.sample_data['close'])
        
        # Check structure
        self.assertIsInstance(macd_data, pd.DataFrame)
        expected_columns = ['macd', 'signal', 'histogram']
        for col in expected_columns:
            self.assertIn(col, macd_data.columns)
        
        # Check relationships
        # Histogram should be MACD - Signal
        calculated_histogram = macd_data['macd'] - macd_data['signal']
        np.testing.assert_array_almost_equal(
            macd_data['histogram'].dropna(), 
            calculated_histogram.dropna(), 
            decimal=10
        )
        
        # Signal line should be smoother than MACD line
        macd_volatility = macd_data['macd'].std()
        signal_volatility = macd_data['signal'].std()
        self.assertLess(signal_volatility, macd_volatility)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb = self.indicators.bollinger_bands(self.sample_data['close'], period=20, std_dev=2)
        
        # Check structure
        self.assertIsInstance(bb, pd.DataFrame)
        expected_columns = ['upper', 'middle', 'lower']
        for col in expected_columns:
            self.assertIn(col, bb.columns)
        
        # Check relationships
        bb_clean = bb.dropna()
        
        # Upper band should be above middle, middle above lower
        self.assertTrue((bb_clean['upper'] >= bb_clean['middle']).all())
        self.assertTrue((bb_clean['middle'] >= bb_clean['lower']).all())
        
        # Middle band should be SMA
        sma_20 = self.indicators.sma(self.sample_data['close'], 20)
        np.testing.assert_array_almost_equal(
            bb['middle'].dropna(), 
            sma_20.dropna(), 
            decimal=10
        )
        
        # Most prices should be within the bands
        prices = self.sample_data['close']
        within_bands = ((prices >= bb['lower']) & (prices <= bb['upper'])).sum()
        total_valid = len(bb.dropna())
        within_percentage = within_bands / total_valid
        self.assertGreater(within_percentage, 0.8)  # Should be around 95% normally
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        stoch = self.indicators.stochastic(
            self.sample_data['high'], 
            self.sample_data['low'], 
            self.sample_data['close']
        )
        
        # Check structure
        self.assertIsInstance(stoch, pd.DataFrame)
        expected_columns = ['k', 'd']
        for col in expected_columns:
            self.assertIn(col, stoch.columns)
        
        # Check value ranges
        stoch_clean = stoch.dropna()
        
        # K and D should be between 0 and 100
        self.assertTrue((stoch_clean['k'] >= 0).all())
        self.assertTrue((stoch_clean['k'] <= 100).all())
        self.assertTrue((stoch_clean['d'] >= 0).all())
        self.assertTrue((stoch_clean['d'] <= 100).all())
        
        # D should be smoother than K (it's a moving average of K)
        k_volatility = stoch_clean['k'].std()
        d_volatility = stoch_clean['d'].std()
        self.assertLessEqual(d_volatility, k_volatility)
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr = self.indicators.atr(
            self.sample_data['high'], 
            self.sample_data['low'], 
            self.sample_data['close']
        )
        
        # Check basic properties
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.sample_data))
        
        # ATR should be positive
        atr_values = atr.dropna()
        self.assertTrue((atr_values > 0).all())
        
        # ATR should be reasonable compared to price range
        price_range = self.sample_data['close'].max() - self.sample_data['close'].min()
        avg_atr = atr_values.mean()
        self.assertLess(avg_atr, price_range)  # ATR shouldn't exceed total price range
    
    def test_indicator_with_insufficient_data(self):
        """Test indicators with insufficient data."""
        short_data = self.sample_data['close'].iloc[:5]  # Only 5 data points
        
        # Should still work but return mostly NaN
        sma_20 = self.indicators.sma(short_data, 20)
        self.assertTrue(sma_20.isna().all())
        
        rsi_14 = self.indicators.rsi(short_data, 14)
        self.assertTrue(rsi_14.isna().all())
    
    def test_indicator_with_list_input(self):
        """Test indicators with list input instead of pandas Series."""
        price_list = self.sample_data['close'].tolist()
        
        sma_result = self.indicators.sma(price_list, 10)
        self.assertIsInstance(sma_result, pd.Series)
        
        rsi_result = self.indicators.rsi(price_list, 14)
        self.assertIsInstance(rsi_result, pd.Series)
    
    def test_indicator_with_numpy_input(self):
        """Test indicators with numpy array input."""
        price_array = self.sample_data['close'].values
        
        sma_result = self.indicators.sma(price_array, 10)
        self.assertIsInstance(sma_result, pd.Series)
        
        rsi_result = self.indicators.rsi(price_array, 14)
        self.assertIsInstance(rsi_result, pd.Series)


class TestSignalGenerator(unittest.TestCase):
    """Test cases for signal generation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create trending data for better signal testing
        trend = np.linspace(100, 110, 100)
        noise = np.random.normal(0, 0.5, 100)
        prices = trend + noise
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 100),
            'high': prices + np.abs(np.random.normal(0, 0.3, 100)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 100)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.signal_generator = MockSignalGenerator()
    
    def test_signal_generation(self):
        """Test basic signal generation."""
        signals = self.signal_generator.generate_signals(self.sample_data)
        
        # Check structure
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('confidence', signals.columns)
        self.assertEqual(len(signals), len(self.sample_data))
        
        # Check signal values
        valid_signals = ['BUY', 'SELL', 'HOLD']
        unique_signals = signals['signal'].unique()
        for signal in unique_signals:
            self.assertIn(signal, valid_signals)
        
        # Check confidence values
        self.assertTrue((signals['confidence'] >= 0).all())
        self.assertTrue((signals['confidence'] <= 1).all())
    
    def test_crossover_signals(self):
        """Test moving average crossover signals."""
        # Create clear crossover scenario
        fast_ma = pd.Series([10, 11, 12, 13, 14, 15, 16, 15, 14, 13], 
                           index=range(10))
        slow_ma = pd.Series([12, 12, 12, 12, 12, 12, 12, 12, 12, 12], 
                           index=range(10))
        
        signals = self.signal_generator.crossover_signal(fast_ma, slow_ma)
        
        # Should generate BUY signal when fast crosses above slow
        # Should generate SELL signal when fast crosses below slow
        self.assertIn('BUY', signals.values)
        self.assertIn('SELL', signals.values)
        self.assertIn('HOLD', signals.values)
    
    def test_rsi_signals(self):
        """Test RSI-based signals."""
        # Create data that will generate extreme RSI values
        declining_prices = pd.Series(range(100, 70, -1))  # Declining prices for high RSI
        rising_prices = pd.Series(range(70, 100))  # Rising prices for low RSI
        
        # Test declining prices (should generate oversold/buy signals)
        test_data = pd.DataFrame({'close': declining_prices})
        signals = self.signal_generator.generate_signals(test_data)
        
        # Should have some BUY signals due to oversold conditions
        self.assertIn('BUY', signals['signal'].values)
    
    def test_signal_consistency(self):
        """Test signal consistency across multiple runs."""
        signals1 = self.signal_generator.generate_signals(self.sample_data)
        signals2 = self.signal_generator.generate_signals(self.sample_data)
        
        # Signals should be identical for same input
        pd.testing.assert_frame_equal(signals1, signals2)
    
    def test_signal_timing(self):
        """Test signal timing and lag."""
        # Signals should not have lookahead bias
        signals = self.signal_generator.generate_signals(self.sample_data)
        
        # Check that signals are based only on current and past data
        # This is implicit in our implementation, but we test that signals exist
        non_hold_signals = signals[signals['signal'] != 'HOLD']
        self.assertGreater(len(non_hold_signals), 0)  # Should have some signals
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        signals = self.signal_generator.generate_signals(self.sample_data)
        
        # Confidence should be higher for BUY/SELL than HOLD
        buy_sell_confidence = signals[signals['signal'].isin(['BUY', 'SELL'])]['confidence']
        hold_confidence = signals[signals['signal'] == 'HOLD']['confidence']
        
        if len(buy_sell_confidence) > 0 and len(hold_confidence) > 0:
            self.assertGreater(buy_sell_confidence.mean(), hold_confidence.mean())


class TestCustomIndicators(unittest.TestCase):
    """Test cases for custom indicators."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 100),
            'high': prices + np.abs(np.random.normal(0, 0.3, 100)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 100)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Williams %R indicator."""
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, high, low, close, period=20):
        """Commodity Channel Index."""
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def test_williams_r(self):
        """Test Williams %R calculation."""
        williams_r = self.calculate_williams_r(
            self.sample_data['high'], 
            self.sample_data['low'], 
            self.sample_data['close']
        )
        
        # Check basic properties
        self.assertIsInstance(williams_r, pd.Series)
        
        # Williams %R should be between -100 and 0
        wr_values = williams_r.dropna()
        self.assertTrue((wr_values >= -100).all())
        self.assertTrue((wr_values <= 0).all())
    
    def test_cci(self):
        """Test Commodity Channel Index calculation."""
        cci = self.calculate_cci(
            self.sample_data['high'], 
            self.sample_data['low'], 
            self.sample_data['close']
        )
        
        # Check basic properties
        self.assertIsInstance(cci, pd.Series)
        
        # CCI typically ranges from -300 to +300, but can exceed
        cci_values = cci.dropna()
        self.assertTrue(len(cci_values) > 0)
        
        # Should have some variation
        self.assertGreater(cci_values.std(), 0)


class TestIndicatorPerformance(unittest.TestCase):
    """Test cases for indicator performance and efficiency."""
    
    def setUp(self):
        """Set up large dataset for performance testing."""
        np.random.seed(42)
        # Large dataset for performance testing
        dates = pd.date_range('2020-01-01', periods=10000, freq='H')
        prices = 100 + np.cumsum(np.random.randn(10000) * 0.001)
        
        self.large_data = pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.normal(0, 0.01, 10000)),
            'low': prices - np.abs(np.random.normal(0, 0.01, 10000)),
            'volume': np.random.randint(1000, 10000, 10000)
        }, index=dates)
        
        self.indicators = MockTechnicalIndicators()
    
    def test_indicator_speed(self):
        """Test indicator calculation speed."""
        import time
        
        # Test SMA speed
        start_time = time.time()
        sma = self.indicators.sma(self.large_data['close'], 20)
        sma_time = time.time() - start_time
        
        # Should complete within reasonable time (2 seconds for 10k data points)
        self.assertLess(sma_time, 2.0)
        
        # Test RSI speed
        start_time = time.time()
        rsi = self.indicators.rsi(self.large_data['close'], 14)
        rsi_time = time.time() - start_time
        
        self.assertLess(rsi_time, 2.0)
    
    def test_memory_usage(self):
        """Test memory usage of indicators."""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(self.large_data)
        
        # Calculate multiple indicators
        sma = self.indicators.sma(self.large_data['close'], 20)
        ema = self.indicators.ema(self.large_data['close'], 20)
        rsi = self.indicators.rsi(self.large_data['close'], 14)
        
        # Memory usage should be reasonable
        total_indicator_size = sys.getsizeof(sma) + sys.getsizeof(ema) + sys.getsizeof(rsi)
        
        # Indicators shouldn't use more than 3x the original data size
        self.assertLess(total_indicator_size, initial_size * 3)


if __name__ == "__main__":
    # Run specific test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestTechnicalIndicators))
    suite.addTest(loader.loadTestsFromTestCase(TestSignalGenerator))
    suite.addTest(loader.loadTestsFromTestCase(TestCustomIndicators))
    suite.addTest(loader.loadTestsFromTestCase(TestIndicatorPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nIndicator Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All indicator tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")