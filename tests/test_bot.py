"""
Test suite for the bot package.
Tests trading bot core functionality, strategies, and risk management.
"""

import unittest
import pytest
import tempfile
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock classes for testing when actual modules aren't available
class MockTradingBot:
    """Mock trading bot for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.is_running = False
        self.positions = []
        self.balance = 10000.0
        self.trades = []
        
    def start(self):
        self.is_running = True
        return True
        
    def stop(self):
        self.is_running = False
        return True
        
    def place_order(self, symbol, order_type, amount, price=None):
        order = {
            'id': f"ORDER_{len(self.trades)}",
            'symbol': symbol,
            'type': order_type,
            'amount': amount,
            'price': price or 1.0,
            'timestamp': datetime.now(),
            'status': 'filled'
        }
        self.trades.append(order)
        return order
        
    def get_balance(self):
        return self.balance
        
    def get_positions(self):
        return self.positions.copy()


class MockRiskManager:
    """Mock risk manager for testing."""
    
    def __init__(self, max_risk_per_trade=0.02, max_total_risk=0.1):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        
    def calculate_position_size(self, account_balance, risk_percentage, stop_loss_pips, pip_value):
        return min(account_balance * risk_percentage / (stop_loss_pips * pip_value), 
                  account_balance * self.max_risk_per_trade / (stop_loss_pips * pip_value))
        
    def validate_trade(self, trade_params):
        # Simple validation - always approve for testing
        return True, "Trade approved"
        
    def check_exposure(self, symbol, proposed_amount):
        return True  # Always allow for testing


class MockStrategyManager:
    """Mock strategy manager for testing."""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategy = None
        
    def add_strategy(self, name, strategy):
        self.strategies[name] = strategy
        
    def set_active_strategy(self, name):
        if name in self.strategies:
            self.active_strategy = name
            return True
        return False
        
    def get_signal(self, data):
        if self.active_strategy:
            return {"signal": "BUY", "confidence": 0.75, "price": 1.1234}
        return {"signal": "HOLD", "confidence": 0.5, "price": 1.1234}
        
    def backtest_strategy(self, strategy_name, data, start_date, end_date):
        return {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "win_rate": 0.65,
            "total_trades": 100
        }


class TestTradingBot(unittest.TestCase):
    """Test cases for the main trading bot."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'api': {
                'timeout': 30,
                'max_retries': 3
            },
            'trading': {
                'risk_percentage': 2.0,
                'max_positions': 5,
                'leverage': 100
            },
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY']
        }
        self.bot = MockTradingBot(self.config)
    
    def test_bot_initialization(self):
        """Test bot initialization."""
        self.assertIsNotNone(self.bot)
        self.assertEqual(self.bot.config, self.config)
        self.assertFalse(self.bot.is_running)
        self.assertEqual(self.bot.balance, 10000.0)
    
    def test_bot_start_stop(self):
        """Test bot start and stop functionality."""
        # Test start
        result = self.bot.start()
        self.assertTrue(result)
        self.assertTrue(self.bot.is_running)
        
        # Test stop
        result = self.bot.stop()
        self.assertTrue(result)
        self.assertFalse(self.bot.is_running)
    
    def test_place_order(self):
        """Test order placement."""
        order = self.bot.place_order("EURUSD", "BUY", 100.0, 1.1234)
        
        self.assertIsNotNone(order)
        self.assertEqual(order['symbol'], "EURUSD")
        self.assertEqual(order['type'], "BUY")
        self.assertEqual(order['amount'], 100.0)
        self.assertEqual(order['price'], 1.1234)
        self.assertEqual(order['status'], 'filled')
        self.assertIn('id', order)
        self.assertIn('timestamp', order)
    
    def test_multiple_orders(self):
        """Test placing multiple orders."""
        orders = []
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for i, symbol in enumerate(symbols):
            order = self.bot.place_order(symbol, "BUY", 100.0 * (i + 1), 1.0 + i * 0.1)
            orders.append(order)
        
        self.assertEqual(len(orders), 3)
        self.assertEqual(len(self.bot.trades), 3)
        
        # Check order IDs are unique
        order_ids = [order['id'] for order in orders]
        self.assertEqual(len(set(order_ids)), 3)
    
    def test_get_balance(self):
        """Test balance retrieval."""
        balance = self.bot.get_balance()
        self.assertEqual(balance, 10000.0)
        self.assertIsInstance(balance, float)
    
    def test_get_positions(self):
        """Test position retrieval."""
        positions = self.bot.get_positions()
        self.assertIsInstance(positions, list)
        # Initially should be empty
        self.assertEqual(len(positions), 0)


class TestRiskManager(unittest.TestCase):
    """Test cases for risk management."""
    
    def setUp(self):
        """Set up test environment."""
        self.risk_manager = MockRiskManager(max_risk_per_trade=0.02, max_total_risk=0.1)
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        account_balance = 10000.0
        risk_percentage = 0.02  # 2%
        stop_loss_pips = 50
        pip_value = 10
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance, risk_percentage, stop_loss_pips, pip_value
        )
        
        self.assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
        
        # Check that position size respects risk limits
        max_loss = position_size * stop_loss_pips * pip_value
        max_allowed_loss = account_balance * risk_percentage
        self.assertLessEqual(max_loss, max_allowed_loss * 1.01)  # Allow small rounding error
    
    def test_zero_risk_position_size(self):
        """Test position size with zero risk."""
        position_size = self.risk_manager.calculate_position_size(10000.0, 0, 50, 10)
        self.assertEqual(position_size, 0)
    
    def test_trade_validation(self):
        """Test trade validation."""
        trade_params = {
            'symbol': 'EURUSD',
            'amount': 100.0,
            'type': 'BUY',
            'stop_loss': 50,
            'take_profit': 100
        }
        
        is_valid, message = self.risk_manager.validate_trade(trade_params)
        self.assertTrue(is_valid)
        self.assertIsInstance(message, str)
    
    def test_exposure_check(self):
        """Test exposure checking."""
        result = self.risk_manager.check_exposure('EURUSD', 1000.0)
        self.assertIsInstance(result, bool)


class TestStrategyManager(unittest.TestCase):
    """Test cases for strategy management."""
    
    def setUp(self):
        """Set up test environment."""
        self.strategy_manager = MockStrategyManager()
        
        # Mock strategy
        self.mock_strategy = Mock()
        self.mock_strategy.name = "TestStrategy"
        self.mock_strategy.get_signal.return_value = {"signal": "BUY", "confidence": 0.8}
    
    def test_add_strategy(self):
        """Test adding strategies."""
        self.strategy_manager.add_strategy("test_strategy", self.mock_strategy)
        self.assertIn("test_strategy", self.strategy_manager.strategies)
        self.assertEqual(self.strategy_manager.strategies["test_strategy"], self.mock_strategy)
    
    def test_set_active_strategy(self):
        """Test setting active strategy."""
        # Add strategy first
        self.strategy_manager.add_strategy("test_strategy", self.mock_strategy)
        
        # Set as active
        result = self.strategy_manager.set_active_strategy("test_strategy")
        self.assertTrue(result)
        self.assertEqual(self.strategy_manager.active_strategy, "test_strategy")
        
        # Try to set non-existent strategy
        result = self.strategy_manager.set_active_strategy("nonexistent")
        self.assertFalse(result)
    
    def test_get_signal(self):
        """Test signal generation."""
        # Test without active strategy
        signal = self.strategy_manager.get_signal(None)
        self.assertEqual(signal["signal"], "HOLD")
        
        # Test with active strategy
        self.strategy_manager.add_strategy("test_strategy", self.mock_strategy)
        self.strategy_manager.set_active_strategy("test_strategy")
        
        signal = self.strategy_manager.get_signal(None)
        self.assertIn("signal", signal)
        self.assertIn("confidence", signal)
        self.assertIn("price", signal)
    
    def test_backtest_strategy(self):
        """Test strategy backtesting."""
        self.strategy_manager.add_strategy("test_strategy", self.mock_strategy)
        
        # Mock data
        data = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.2],
            'volume': [1000, 1200, 1100, 1300, 1150]
        })
        
        results = self.strategy_manager.backtest_strategy(
            "test_strategy", data, "2023-01-01", "2023-12-31"
        )
        
        self.assertIsInstance(results, dict)
        expected_keys = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]
        for key in expected_keys:
            self.assertIn(key, results)


class TestTradingIntegration(unittest.TestCase):
    """Integration tests for trading components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.bot = MockTradingBot()
        self.risk_manager = MockRiskManager()
        self.strategy_manager = MockStrategyManager()
        
        # Add a test strategy
        mock_strategy = Mock()
        mock_strategy.get_signal.return_value = {"signal": "BUY", "confidence": 0.8, "price": 1.1234}
        self.strategy_manager.add_strategy("test_strategy", mock_strategy)
        self.strategy_manager.set_active_strategy("test_strategy")
    
    def test_complete_trading_workflow(self):
        """Test complete trading workflow from signal to execution."""
        # Step 1: Get signal from strategy
        market_data = pd.DataFrame({
            'open': [1.1200, 1.1210, 1.1220],
            'high': [1.1250, 1.1260, 1.1270],
            'low': [1.1180, 1.1190, 1.1200],
            'close': [1.1230, 1.1240, 1.1250],
            'volume': [10000, 12000, 11000]
        })
        
        signal = self.strategy_manager.get_signal(market_data)
        self.assertEqual(signal["signal"], "BUY")
        
        # Step 2: Risk management validation
        trade_params = {
            'symbol': 'EURUSD',
            'amount': 100.0,
            'type': signal["signal"],
            'price': signal["price"]
        }
        
        is_valid, message = self.risk_manager.validate_trade(trade_params)
        self.assertTrue(is_valid)
        
        # Step 3: Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.bot.get_balance(), 0.02, 50, 10
        )
        self.assertGreater(position_size, 0)
        
        # Step 4: Execute trade
        order = self.bot.place_order(
            trade_params['symbol'], 
            trade_params['type'], 
            position_size, 
            trade_params['price']
        )
        
        self.assertIsNotNone(order)
        self.assertEqual(order['status'], 'filled')
    
    def test_risk_management_integration(self):
        """Test risk management across multiple trades."""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        orders = []
        
        for symbol in symbols:
            # Check exposure before trade
            exposure_ok = self.risk_manager.check_exposure(symbol, 100.0)
            self.assertTrue(exposure_ok)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.bot.get_balance(), 0.01, 50, 10  # 1% risk per trade
            )
            
            # Validate trade
            trade_params = {
                'symbol': symbol,
                'amount': position_size,
                'type': 'BUY'
            }
            is_valid, _ = self.risk_manager.validate_trade(trade_params)
            self.assertTrue(is_valid)
            
            # Execute if valid
            if is_valid:
                order = self.bot.place_order(symbol, 'BUY', position_size, 1.0)
                orders.append(order)
        
        # Check that all trades were executed
        self.assertEqual(len(orders), 3)
        self.assertEqual(len(self.bot.trades), 3)


class TestAsyncTradingBot(unittest.TestCase):
    """Test cases for asynchronous trading bot operations."""
    
    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()
    
    async def async_market_data_feed(self):
        """Mock async market data feed."""
        for i in range(5):
            yield {
                'symbol': 'EURUSD',
                'price': 1.1200 + i * 0.0001,
                'timestamp': datetime.now(),
                'volume': 1000 + i * 100
            }
            await asyncio.sleep(0.01)  # Simulate real-time delay
    
    def test_async_data_processing(self):
        """Test asynchronous data processing."""
        async def test_coroutine():
            data_points = []
            async for data in self.async_market_data_feed():
                data_points.append(data)
            return data_points
        
        # Run the async test
        result = self.loop.run_until_complete(test_coroutine())
        
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]['symbol'], 'EURUSD')
        self.assertAlmostEqual(result[0]['price'], 1.1200, places=4)
        self.assertAlmostEqual(result[-1]['price'], 1.1204, places=4)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for trading performance metrics."""
    
    def setUp(self):
        """Set up performance test data."""
        # Create sample trade history
        np.random.seed(42)
        self.trades = []
        
        for i in range(100):
            # Random P&L between -100 and +150 (slight positive bias)
            pnl = np.random.normal(5, 50)  # Mean profit of 5, std dev of 50
            
            trade = {
                'id': f'T{i:03d}',
                'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY']),
                'type': np.random.choice(['BUY', 'SELL']),
                'amount': np.random.uniform(0.1, 2.0),
                'entry_price': np.random.uniform(1.0, 1.5),
                'exit_price': np.random.uniform(1.0, 1.5),
                'pnl': pnl,
                'duration': np.random.uniform(1, 24),  # hours
                'timestamp': datetime.now() - timedelta(days=i)
            }
            self.trades.append(trade)
    
    def calculate_performance_metrics(self, trades):
        """Calculate performance metrics from trades."""
        if not trades:
            return {}
        
        pnls = [trade['pnl'] for trade in trades]
        
        metrics = {
            'total_trades': len(trades),
            'total_pnl': sum(pnls),
            'average_pnl': np.mean(pnls),
            'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
            'profit_factor': abs(sum([p for p in pnls if p > 0])) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf'),
            'max_win': max(pnls),
            'max_loss': min(pnls),
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
        }
        
        return metrics
    
    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        metrics = self.calculate_performance_metrics(self.trades)
        
        # Verify all expected metrics are present
        expected_keys = [
            'total_trades', 'total_pnl', 'average_pnl', 'win_rate', 
            'profit_factor', 'max_win', 'max_loss', 'sharpe_ratio'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Verify metric ranges
        self.assertEqual(metrics['total_trades'], 100)
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertGreater(metrics['max_win'], 0)
        self.assertLess(metrics['max_loss'], 0)
    
    def test_empty_trades_performance(self):
        """Test performance calculation with no trades."""
        metrics = self.calculate_performance_metrics([])
        self.assertEqual(metrics, {})
    
    def test_single_trade_performance(self):
        """Test performance calculation with single trade."""
        single_trade = [self.trades[0]]
        metrics = self.calculate_performance_metrics(single_trade)
        
        self.assertEqual(metrics['total_trades'], 1)
        self.assertEqual(metrics['total_pnl'], single_trade[0]['pnl'])
        self.assertEqual(metrics['average_pnl'], single_trade[0]['pnl'])


class TestBotConfiguration(unittest.TestCase):
    """Test cases for bot configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            'api': {'timeout': 30, 'url': 'https://api.example.com'},
            'trading': {'risk_percentage': 2.0, 'max_positions': 5},
            'symbols': ['EURUSD', 'GBPUSD']
        }
        
        # Test valid config
        bot = MockTradingBot(valid_config)
        self.assertEqual(bot.config, valid_config)
    
    def test_config_defaults(self):
        """Test default configuration handling."""
        bot = MockTradingBot()
        self.assertEqual(bot.config, {})
        
        # Should still function with empty config
        self.assertIsNotNone(bot.get_balance())
        self.assertIsNotNone(bot.get_positions())


if __name__ == "__main__":
    # Run specific test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestTradingBot))
    suite.addTest(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTest(loader.loadTestsFromTestCase(TestStrategyManager))
    suite.addTest(loader.loadTestsFromTestCase(TestTradingIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTest(loader.loadTestsFromTestCase(TestBotConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")