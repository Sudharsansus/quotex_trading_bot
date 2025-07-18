"""
Market Data Handler
Real-time and historical market data management
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque
import json

from .quotex_api import quotex_api

logger = logging.getLogger(__name__)

class MarketDataManager:
    def __init__(self, max_history_length: int = 1000):
        self.max_history_length = max_history_length
        self.candle_data: Dict[str, deque] = {}
        self.tick_data: Dict[str, deque] = {}
        self.subscribed_assets: set = set()
        self.data_callbacks: Dict[str, List] = {}
        
    async def initialize(self):
        """Initialize market data manager"""
        try:
            # Set up event handlers
            quotex_api.on_candle_update = self._on_candle_update
            logger.info("Market data manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize market data manager: {e}")
    
    async def subscribe_to_asset(self, asset: str, timeframe: int = 60):
        """Subscribe to real-time data for an asset"""
        try:
            if asset not in self.subscribed_assets:
                await quotex_api.subscribe_to_asset(asset, timeframe)
                self.subscribed_assets.add(asset)
                
                # Initialize data storage
                if asset not in self.candle_data:
                    self.candle_data[asset] = deque(maxlen=self.max_history_length)
                    self.tick_data[asset] = deque(maxlen=self.max_history_length)
                
                # Load historical data
                await self._load_historical_data(asset, timeframe)
                
                logger.info(f"Subscribed to {asset} with timeframe {timeframe}")
                
        except Exception as e:
            logger.error(f"Failed to subscribe to {asset}: {e}")
    
    async def unsubscribe_from_asset(self, asset: str):
        """Unsubscribe from real-time data for an asset"""
        try:
            if asset in self.subscribed_assets:
                await quotex_api.unsubscribe_from_asset(asset)
                self.subscribed_assets.remove(asset)
                logger.info(f"Unsubscribed from {asset}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {asset}: {e}")
    
    async def _load_historical_data(self, asset: str, timeframe: int, count: int = 100):
        """Load historical candle data"""
        try:
            candles = await quotex_api.get_candles(asset, timeframe, count)
            
            for candle in candles:
                processed_candle = self._process_candle(candle)
                self.candle_data[asset].append(processed_candle)
            
            logger.info(f"Loaded {len(candles)} historical candles for {asset}")
            
        except Exception as e:
            logger.error(f"Failed to load historical data for {asset}: {e}")
    
    async def _on_candle_update(self, data: dict):
        """Handle real-time candle updates"""
        try:
            asset = data.get('asset')
            if asset and asset in self.subscribed_assets:
                processed_candle = self._process_candle(data)
                self.candle_data[asset].append(processed_candle)
                
                # Trigger callbacks
                if asset in self.data_callbacks:
                    for callback in self.data_callbacks[asset]:
                        await callback(asset, processed_candle)
                        
        except Exception as e:
            logger.error(f"Error processing candle update: {e}")
    
    def _process_candle(self, candle_data: dict) -> dict:
        """Process raw candle data"""
        return {
            'timestamp': candle_data.get('timestamp', datetime.now().timestamp()),
            'open': float(candle_data.get('open', 0)),
            'high': float(candle_data.get('high', 0)),
            'low': float(candle_data.get('low', 0)),
            'close': float(candle_data.get('close', 0)),
            'volume': float(candle_data.get('volume', 0)),
            'asset': candle_data.get('asset', ''),
            'timeframe': candle_data.get('timeframe', 60)
        }
    
    def get_candle_data(self, asset: str, count: int = 100) -> pd.DataFrame:
        """Get candle data as DataFrame"""
        try:
            if asset not in self.candle_data:
                return pd.DataFrame()
            
            candles = list(self.candle_data[asset])[-count:]
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get candle data for {asset}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, asset: str) -> float:
        """Get latest price for an asset"""
        try:
            if asset in self.candle_data and self.candle_data[asset]:
                return self.candle_data[asset][-1]['close']
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get latest price for {asset}: {e}")
            return 0.0
    
    def get_price_change(self, asset: str, periods: int = 1) -> float:
        """Get price change over specified periods"""
        try:
            if asset not in self.candle_data or len(self.candle_data[asset]) < periods + 1:
                return 0.0
            
            current_price = self.candle_data[asset][-1]['close']
            previous_price = self.candle_data[asset][-periods-1]['close']
            
            return ((current_price - previous_price) / previous_price) * 100
            
        except Exception as e:
            logger.error(f"Failed to get price change for {asset}: {e}")
            return 0.0
    
    def get_volatility(self, asset: str, periods: int = 20) -> float:
        """Calculate volatility for an asset"""
        try:
            df = self.get_candle_data(asset, periods)
            if df.empty:
                return 0.0
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(periods)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility for {asset}: {e}")
            return 0.0
    
    def get_support_resistance(self, asset: str, periods: int = 50) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            df = self.get_candle_data(asset, periods)
            if df.empty:
                return 0.0, 0.0
            
            # Simple support/resistance calculation
            support = df['low'].rolling(window=10).min().iloc[-1]
            resistance = df['high'].rolling(window=10).max().iloc[-1]
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Failed to calculate support/resistance for {asset}: {e}")
            return 0.0, 0.0
    
    def add_data_callback(self, asset: str, callback):
        """Add callback for real-time data updates"""
        if asset not in self.data_callbacks:
            self.data_callbacks[asset] = []
        self.data_callbacks[asset].append(callback)
    
    def remove_data_callback(self, asset: str, callback):
        """Remove callback for real-time data updates"""
        if asset in self.data_callbacks:
            try:
                self.data_callbacks[asset].remove(callback)
            except ValueError:
                pass
    
    def get_market_summary(self) -> dict:
        """Get market summary for all subscribed assets"""
        summary = {}
        
        for asset in self.subscribed_assets:
            try:
                latest_price = self.get_latest_price(asset)
                price_change = self.get_price_change(asset)
                volatility = self.get_volatility(asset)
                support, resistance = self.get_support_resistance(asset)
                
                summary[asset] = {
                    'price': latest_price,
                    'change_pct': price_change,
                    'volatility': volatility,
                    'support': support,
                    'resistance': resistance,
                    'data_points': len(self.candle_data.get(asset, []))
                }
                
            except Exception as e:
                logger.error(f"Failed to get summary for {asset}: {e}")
                summary[asset] = {
                    'price': 0.0,
                    'change_pct': 0.0,
                    'volatility': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                    'data_points': 0
                }
        
        return summary
    
    def export_data(self, asset: str, filepath: str):
        """Export asset data to CSV"""
        try:
            df = self.get_candle_data(asset)
            if not df.empty:
                df.to_csv(filepath)
                logger.info(f"Exported {asset} data to {filepath}")
            else:
                logger.warning(f"No data available for {asset}")
        except Exception as e:
            logger.error(f"Failed to export data for {asset}: {e}")
    
    def clear_data(self, asset: str = None):
        """Clear data for specific asset or all assets"""
        try:
            if asset:
                if asset in self.candle_data:
                    self.candle_data[asset].clear()
                if asset in self.tick_data:
                    self.tick_data[asset].clear()
            else:
                self.candle_data.clear()
                self.tick_data.clear()
            
            logger.info(f"Cleared data for {asset if asset else 'all assets'}")
            
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Unsubscribe from all assets
            for asset in list(self.subscribed_assets):
                await self.unsubscribe_from_asset(asset)
            
            # Clear all data
            self.clear_data()
            
            logger.info("Market data manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Singleton instance
market_data_manager = MarketDataManager()