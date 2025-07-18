"""
Quotex API Interface
Complete API wrapper for Quotex trading platform
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import websockets
from urllib.parse import urlencode
import hashlib
import hmac
import base64

from config.credentials import get_quotex_credentials, QUOTEX_CONFIG, TRADING_MODE

logger = logging.getLogger(__name__)

class QuotexAPI:
    def __init__(self):
        self.session = None
        self.ws_connection = None
        self.is_connected = False
        self.is_authenticated = False
        self.credentials = get_quotex_credentials()
        self.user_id = None
        self.session_token = None
        self.balance = 0
        self.demo_mode = TRADING_MODE['demo']
        
        # Event handlers
        self.on_candle_update = None
        self.on_order_update = None
        self.on_balance_update = None
        
    async def connect(self) -> bool:
        """Connect to Quotex API"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=QUOTEX_CONFIG['timeout'])
            )
            
            # Login to get session token
            login_success = await self._login()
            if not login_success:
                return False
            
            # Connect to WebSocket
            await self._connect_websocket()
            
            self.is_connected = True
            logger.info("Successfully connected to Quotex API")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Quotex API: {e}")
            return False
    
    async def _login(self) -> bool:
        """Login to Quotex platform"""
        try:
            login_data = {
                'email': self.credentials['email'],
                'password': self.credentials['password'],
                'platform': 'web'
            }
            
            async with self.session.post(
                f"{QUOTEX_CONFIG['api_url']}/login",
                json=login_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.session_token = data.get('token')
                    self.user_id = data.get('user_id')
                    self.is_authenticated = True
                    
                    # Set session headers
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.session_token}',
                        'Content-Type': 'application/json'
                    })
                    
                    logger.info("Successfully authenticated with Quotex")
                    return True
                else:
                    logger.error(f"Login failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    async def _connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        try:
            headers = {
                'Authorization': f'Bearer {self.session_token}'
            }
            
            self.ws_connection = await websockets.connect(
                QUOTEX_CONFIG['websocket_url'],
                extra_headers=headers
            )
            
            # Start listening for messages
            asyncio.create_task(self._ws_message_handler())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _ws_message_handler(self):
        """Handle WebSocket messages"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                await self._process_ws_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket message handler error: {e}")
    
    async def _process_ws_message(self, data: dict):
        """Process WebSocket messages"""
        message_type = data.get('type')
        
        if message_type == 'candle':
            if self.on_candle_update:
                await self.on_candle_update(data)
        
        elif message_type == 'order':
            if self.on_order_update:
                await self.on_order_update(data)
        
        elif message_type == 'balance':
            self.balance = data.get('balance', 0)
            if self.on_balance_update:
                await self.on_balance_update(data)
    
    async def get_balance(self) -> float:
        """Get current account balance"""
        try:
            async with self.session.get(
                f"{QUOTEX_CONFIG['api_url']}/balance"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    balance_type = 'demo' if self.demo_mode else 'live'
                    self.balance = data.get(balance_type, 0)
                    return self.balance
                return 0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    async def get_candles(self, asset: str, timeframe: int, count: int = 100) -> List[dict]:
        """Get historical candle data"""
        try:
            params = {
                'asset': asset,
                'timeframe': timeframe,
                'count': count,
                'timestamp': int(time.time())
            }
            
            async with self.session.get(
                f"{QUOTEX_CONFIG['api_url']}/candles",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('candles', [])
                return []
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            return []
    
    async def place_order(self, asset: str, direction: str, amount: float, 
                         duration: int = 60) -> dict:
        """Place a binary options order"""
        try:
            order_data = {
                'asset': asset,
                'direction': direction,  # 'call' or 'put'
                'amount': amount,
                'duration': duration,
                'type': 'binary',
                'demo': self.demo_mode
            }
            
            async with self.session.post(
                f"{QUOTEX_CONFIG['api_url']}/orders",
                json=order_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Order placed: {data}")
                    return data
                else:
                    logger.error(f"Order placement failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {}
    
    async def get_assets(self) -> List[dict]:
        """Get available trading assets"""
        try:
            async with self.session.get(
                f"{QUOTEX_CONFIG['api_url']}/assets"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('assets', [])
                return []
        except Exception as e:
            logger.error(f"Failed to get assets: {e}")
            return []
    
    async def get_orders(self, limit: int = 50) -> List[dict]:
        """Get order history"""
        try:
            params = {'limit': limit}
            
            async with self.session.get(
                f"{QUOTEX_CONFIG['api_url']}/orders",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('orders', [])
                return []
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            async with self.session.delete(
                f"{QUOTEX_CONFIG['api_url']}/orders/{order_id}"
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def subscribe_to_asset(self, asset: str, timeframe: int = 60):
        """Subscribe to real-time asset data"""
        try:
            if self.ws_connection:
                subscription_data = {
                    'type': 'subscribe',
                    'asset': asset,
                    'timeframe': timeframe
                }
                await self.ws_connection.send(json.dumps(subscription_data))
        except Exception as e:
            logger.error(f"Failed to subscribe to asset: {e}")
    
    async def unsubscribe_from_asset(self, asset: str):
        """Unsubscribe from real-time asset data"""
        try:
            if self.ws_connection:
                subscription_data = {
                    'type': 'unsubscribe',
                    'asset': asset
                }
                await self.ws_connection.send(json.dumps(subscription_data))
        except Exception as e:
            logger.error(f"Failed to unsubscribe from asset: {e}")
    
    async def get_market_status(self) -> dict:
        """Get market status information"""
        try:
            async with self.session.get(
                f"{QUOTEX_CONFIG['api_url']}/market-status"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {}
    
    async def disconnect(self):
        """Disconnect from Quotex API"""
        try:
            if self.ws_connection:
                await self.ws_connection.close()
            
            if self.session:
                await self.session.close()
            
            self.is_connected = False
            self.is_authenticated = False
            logger.info("Disconnected from Quotex API")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def set_demo_mode(self, demo: bool):
        """Switch between demo and live trading"""
        self.demo_mode = demo
        logger.info(f"Trading mode set to: {'Demo' if demo else 'Live'}")
    
    def is_market_open(self, asset: str) -> bool:
        """Check if market is open for specific asset"""
        # This would typically check market hours
        # For now, return True (implement based on actual market hours)
        return True
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# Singleton instance
quotex_api = QuotexAPI()