"""
Risk Management System for Quotex Trading Bot
Handles position sizing, stop-loss, drawdown protection, and risk metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class TradeType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    current_streak: int = 0
    max_losing_streak: int = 0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    expected_shortfall: float = 0.0

@dataclass
class TradeRecord:
    """Individual trade record for risk calculation"""
    timestamp: datetime
    symbol: str
    trade_type: TradeType
    amount: float
    pnl: float
    duration: int  # in minutes
    win: bool

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 100.0  # Maximum position size
    max_daily_loss: float = 500.0     # Maximum daily loss
    max_weekly_loss: float = 2000.0   # Maximum weekly loss
    max_monthly_loss: float = 5000.0  # Maximum monthly loss
    max_drawdown_percent: float = 20.0  # Maximum drawdown percentage
    max_consecutive_losses: int = 5   # Maximum consecutive losses
    max_risk_per_trade: float = 2.0   # Maximum risk per trade as % of balance
    min_balance: float = 1000.0       # Minimum balance to continue trading

class RiskManager:
    """
    Comprehensive risk management system for trading bot
    """
    
    def __init__(self, initial_balance: float = 10000.0, config: Dict = None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Risk limits
        self.risk_limits = RiskLimits()
        if config:
            self._load_config(config)
        
        # Trade history
        self.trade_history: List[TradeRecord] = []
        
        # Risk metrics
        self.risk_metrics = RiskMetrics()
        
        # Internal state
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.trading_enabled = True
        
        # Emergency stop conditions
        self.emergency_stop = False
        self.emergency_reason = ""
        
        logger.info(f"Risk Manager initialized with balance: ${initial_balance}")
    
    def _load_config(self, config: Dict):
        """Load risk configuration from dictionary"""
        risk_config = config.get('risk_management', {})
        
        for key, value in risk_config.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
    
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss: float, 
                              risk_percent: float = None) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_percent: Risk percentage (optional, uses default if None)
        
        Returns:
            Calculated position size
        """
        if not self.trading_enabled or self.emergency_stop:
            return 0.0
        
        # Use default risk percentage if not provided
        if risk_percent is None:
            risk_percent = self.risk_limits.max_risk_per_trade
        
        # Calculate risk per trade in dollars
        risk_amount = self.current_balance * (risk_percent / 100)
        
        # For binary options, position size is the investment amount
        # Risk is typically 80-90% of investment (loss) vs 70-80% profit
        # Assuming 85% loss rate for calculation
        position_size = risk_amount / 0.85
        
        # Apply maximum position size limit
        position_size = min(position_size, self.risk_limits.max_position_size)
        
        # Ensure minimum viable position
        position_size = max(position_size, 1.0)
        
        # Additional risk checks
        position_size = self._apply_risk_adjustments(position_size)
        
        logger.debug(f"Calculated position size: ${position_size:.2f} for {symbol}")
        return position_size
    
    def _apply_risk_adjustments(self, base_position_size: float) -> float:
        """Apply additional risk adjustments based on current conditions"""
        
        adjusted_size = base_position_size
        
        # Reduce size based on consecutive losses
        if self.consecutive_losses > 0:
            reduction_factor = 1 - (self.consecutive_losses * 0.1)
            reduction_factor = max(0.3, reduction_factor)  # Minimum 30% of base size
            adjusted_size *= reduction_factor
        
        # Reduce size based on current drawdown
        if self.risk_metrics.current_drawdown > 10:
            drawdown_factor = 1 - (self.risk_metrics.current_drawdown / 100)
            adjusted_size *= max(0.2, drawdown_factor)
        
        # Reduce size based on daily performance
        if self.risk_metrics.daily_pnl < -100:  # If losing more than $100 today
            adjusted_size *= 0.5
        
        # Increase size after winning streaks (with caution)
        if self.consecutive_wins > 3:
            boost_factor = min(1.2, 1 + (self.consecutive_wins * 0.05))
            adjusted_size *= boost_factor
        
        return adjusted_size
    
    def can_place_trade(self, position_size: float, symbol: str = None) -> Tuple[bool, str]:
        """
        Check if a trade can be placed based on risk limits
        
        Args:
            position_size: Proposed position size
            symbol: Trading symbol (optional)
        
        Returns:
            Tuple of (can_trade, reason)
        """
        if self.emergency_stop:
            return False, f"Emergency stop active: {self.emergency_reason}"
        
        if not self.trading_enabled:
            return False, "Trading disabled by risk manager"
        
        # Check balance
        if self.current_balance < self.risk_limits.min_balance:
            return False, f"Balance below minimum: ${self.current_balance:.2f}"
        
        # Check position size
        if position_size > self.risk_limits.max_position_size:
            return False, f"Position size too large: ${position_size:.2f}"
        
        # Check daily loss limit
        if abs(self.risk_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: ${abs(self.risk_metrics.daily_pnl):.2f}"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
            return False, f"Maximum consecutive losses reached: {self.consecutive_losses}"
        
        # Check drawdown
        if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_percent:
            return False, f"Maximum drawdown exceeded: {self.risk_metrics.current_drawdown:.2f}%"
        
        # Check if sufficient balance for trade
        if position_size > self.current_balance:
            return False, f"Insufficient balance: ${self.current_balance:.2f}"
        
        return True, "Trade approved"
    
    def record_trade(self, 
                    symbol: str, 
                    trade_type: TradeType, 
                    amount: float, 
                    pnl: float, 
                    duration: int = 60) -> None:
        """
        Record a completed trade and update risk metrics
        
        Args:
            symbol: Trading symbol
            trade_type: Type of trade (CALL/PUT)
            amount: Position size
            pnl: Profit/Loss from the trade
            duration: Trade duration in minutes
        """
        # Create trade record
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            trade_type=trade_type,
            amount=amount,
            pnl=pnl,
            duration=duration,
            win=pnl > 0
        )
        
        # Add to history
        self.trade_history.append(trade)
        
        # Update balance
        self.current_balance += pnl
        
        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update streaks
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update daily trade count
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
        self.daily_trades += 1
        
        # Recalculate risk metrics
        self._update_risk_metrics()
        
        # Check for emergency conditions
        self._check_emergency_conditions()
        
        logger.info(f"Trade recorded: {symbol} {trade_type.value} ${amount:.2f} PnL: ${pnl:.2f}")
    
    def _update_risk_metrics(self) -> None:
        """Update all risk metrics based on trade history"""
        if not self.trade_history:
            return
        
        # Calculate drawdown
        self.risk_metrics.current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
        
        # Calculate maximum drawdown
        peak = self.initial_balance
        max_dd = 0
        for trade in self.trade_history:
            balance = peak + sum(t.pnl for t in self.trade_history[:self.trade_history.index(trade)+1])
            if balance > peak:
                peak = balance
            drawdown = ((peak - balance) / peak) * 100
            if drawdown > max_dd:
                max_dd = drawdown
        self.risk_metrics.max_drawdown = max_dd
        
        # Calculate win rate
        wins = sum(1 for trade in self.trade_history if trade.win)
        total_trades = len(self.trade_history)
        self.risk_metrics.win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.pnl for trade in self.trade_history if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trade_history if trade.pnl < 0))
        self.risk_metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate current streak
        self.risk_metrics.current_streak = max(self.consecutive_wins, -self.consecutive_losses)
        
        # Calculate max losing streak
        max_losing_streak = 0
        current_losing_streak = 0
        for trade in self.trade_history:
            if not trade.win:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0
        self.risk_metrics.max_losing_streak = max_losing_streak
        
        # Calculate period PnL
        self._calculate_period_pnl()
        
        # Calculate VaR and Expected Shortfall
        self._calculate_var_metrics()
    
    def _calculate_period_pnl(self) -> None:
        """Calculate PnL for different time periods"""
        now = datetime.now()
        
        # Daily PnL
        daily_trades = [t for t in self.trade_history if t.timestamp.date() == now.date()]
        self.risk_metrics.daily_pnl = sum(t.pnl for t in daily_trades)
        
        # Weekly PnL
        week_start = now - timedelta(days=7)
        weekly_trades = [t for t in self.trade_history if t.timestamp >= week_start]
        self.risk_metrics.weekly_pnl = sum(t.pnl for t in weekly_trades)
        
        # Monthly PnL
        month_start = now - timedelta(days=30)
        monthly_trades = [t for t in self.trade_history if t.timestamp >= month_start]
        self.risk_metrics.monthly_pnl = sum(t.pnl for t in monthly_trades)
    
    def _calculate_var_metrics(self) -> None:
        """Calculate Value at Risk and Expected Shortfall"""
        if len(self.trade_history) < 20:  # Need sufficient data
            return
        
        # Get daily returns
        daily_returns = []
        current_date = None
        daily_pnl = 0
        
        for trade in sorted(self.trade_history, key=lambda t: t.timestamp):
            trade_date = trade.timestamp.date()
            if current_date != trade_date:
                if current_date is not None:
                    daily_returns.append(daily_pnl)
                current_date = trade_date
                daily_pnl = trade.pnl
            else:
                daily_pnl += trade.pnl
        
        if daily_pnl != 0:  # Add last day
            daily_returns.append(daily_pnl)
        
        if len(daily_returns) < 10:
            return
        
        # Calculate VaR (95% confidence)
        sorted_returns = sorted(daily_returns)
        var_index = int(len(sorted_returns) * 0.05)
        self.risk_metrics.var_95 = abs(sorted_returns[var_index])
        
        # Calculate Expected Shortfall (average of worst 5%)
        worst_returns = sorted_returns[:var_index+1]
        self.risk_metrics.expected_shortfall = abs(np.mean(worst_returns)) if worst_returns else 0
    
    def _check_emergency_conditions(self) -> None:
        """Check for emergency stop conditions"""
        
        # Emergency stop if balance is too low
        if self.current_balance < self.risk_limits.min_balance:
            self.emergency_stop = True
            self.emergency_reason = f"Balance below minimum: ${self.current_balance:.2f}"
            self.trading_enabled = False
            return
        
        # Emergency stop if drawdown is too high
        if self.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_percent:
            self.emergency_stop = True
            self.emergency_reason = f"Maximum drawdown exceeded: {self.risk_metrics.current_drawdown:.2f}%"
            self.trading_enabled = False
            return
        
        # Emergency stop if daily loss is too high
        if abs(self.risk_metrics.daily_pnl) > self.risk_limits.max_daily_loss:
            self.emergency_stop = True
            self.emergency_reason = f"Daily loss limit exceeded: ${abs(self.risk_metrics.daily_pnl):.2f}"
            self.trading_enabled = False
            return
        
        # Emergency stop if too many consecutive losses
        if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
            self.emergency_stop = True
            self.emergency_reason = f"Maximum consecutive losses: {self.consecutive_losses}"
            self.trading_enabled = False
            return
    
    def get_risk_level(self) -> RiskLevel:
        """Determine current risk level based on metrics"""
        
        risk_score = 0
        
        # Drawdown factor
        if self.risk_metrics.current_drawdown > 15:
            risk_score += 3
        elif self.risk_metrics.current_drawdown > 10:
            risk_score += 2
        elif self.risk_metrics.current_drawdown > 5:
            risk_score += 1
        
        # Consecutive losses factor
        if self.consecutive_losses > 4:
            risk_score += 3
        elif self.consecutive_losses > 2:
            risk_score += 2
        elif self.consecutive_losses > 0:
            risk_score += 1
        
        # Win rate factor
        if self.risk_metrics.win_rate < 40:
            risk_score += 2
        elif self.risk_metrics.win_rate < 50:
            risk_score += 1
        
        # Daily PnL factor
        if self.risk_metrics.daily_pnl < -200:
            risk_score += 2
        elif self.risk_metrics.daily_pnl < -100:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 8:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        
        return {
            "balance": {
                "current": self.current_balance,
                "initial": self.initial_balance,
                "peak": self.peak_balance,
                "total_return": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
            },
            "risk_metrics": {
                "current_drawdown": self.risk_metrics.current_drawdown,
                "max_drawdown": self.risk_metrics.max_drawdown,
                "win_rate": self.risk_metrics.win_rate,
                "profit_factor": self.risk_metrics.profit_factor,
                "current_streak": self.risk_metrics.current_streak,
                "max_losing_streak": self.risk_metrics.max_losing_streak,
                "var_95": self.risk_metrics.var_95,
                "expected_shortfall": self.risk_metrics.expected_shortfall
            },
            "period_pnl": {
                "daily": self.risk_metrics.daily_pnl,
                "weekly": self.risk_metrics.weekly_pnl,
                "monthly": self.risk_metrics.monthly_pnl
            },
            "trading_status": {
                "enabled": self.trading_enabled,
                "emergency_stop": self.emergency_stop,
                "emergency_reason": self.emergency_reason,
                "risk_level": self.get_risk_level().value,
                "consecutive_losses": self.consecutive_losses,
                "consecutive_wins": self.consecutive_wins,
                "daily_trades": self.daily_trades
            },
            "limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_drawdown_percent": self.risk_limits.max_drawdown_percent,
                "max_consecutive_losses": self.risk_limits.max_consecutive_losses,
                "max_risk_per_trade": self.risk_limits.max_risk_per_trade
            }
        }
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if conditions are met"""
        
        if not self.emergency_stop:
            return True
        
        # Check if conditions have improved
        if (self.current_balance >= self.risk_limits.min_balance and
            self.risk_metrics.current_drawdown <= self.risk_limits.max_drawdown_percent and
            self.consecutive_losses < self.risk_limits.max_consecutive_losses):
            
            self.emergency_stop = False
            self.emergency_reason = ""
            self.trading_enabled = True
            
            logger.info("Emergency stop reset - trading enabled")
            return True
        
        return False
    
    def adjust_risk_limits(self, new_limits: Dict) -> None:
        """Adjust risk limits dynamically"""
        
        for key, value in new_limits.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                logger.info(f"Risk limit updated: {key} = {value}")
    
    def export_trade_history(self, filename: str = None) -> str:
        """Export trade history to JSON file"""
        
        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        trade_data = []
        for trade in self.trade_history:
            trade_data.append({
                "timestamp": trade.timestamp.isoformat(),
                "symbol": trade.symbol,
                "trade_type": trade.trade_type.value,
                "amount": trade.amount,
                "pnl": trade.pnl,
                "duration": trade.duration,
                "win": trade.win
            })
        
        export_data = {
            "trade_history": trade_data,
            "risk_report": self.get_risk_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Trade history exported to {filename}")
        return filename