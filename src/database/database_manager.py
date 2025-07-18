import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from sqlalchemy import create_engine, func, and_, or_, desc, asc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import logging

from .models import (
    Base, TradingAccount, Asset, Trade, MarketData, TradingSignal,
    Strategy, BacktestResult, MLModel, ModelPrediction, SystemLog, SystemConfig
)

class DatabaseManager:
    """Database manager for the Quotex trading bot"""
    
    def __init__(self, database_url: str = None):
        """Initialize database manager
        
        Args:
            database_url: Database connection URL. If None, uses SQLite
        """
        if database_url is None:
            db_path = os.path.join(os.path.dirname(__file__), '../../data/trading_bot.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger = logging.getLogger(__name__)
        
        # Create tables
        self.init_database()
    
    def init_database(self):
        """Create database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    # Account Management
    def create_account(self, account_data: Dict[str, Any]) -> TradingAccount:
        """Create a new trading account"""
        session = self.get_session()
        try:
            account = TradingAccount(**account_data)
            session.add(account)
            session.commit()
            session.refresh(account)
            return account
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating account: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_account(self, account_id: str) -> Optional[TradingAccount]:
        """Get account by ID"""
        session = self.get_session()
        try:
            return session.query(TradingAccount).filter(
                TradingAccount.account_id == account_id
            ).first()
        finally:
            self.close_session(session)
    
    def update_account_balance(self, account_id: str, balance: float, equity: float = None):
        """Update account balance"""
        session = self.get_session()
        try:
            account = session.query(TradingAccount).filter(
                TradingAccount.account_id == account_id
            ).first()
            if account:
                account.balance = balance
                if equity is not None:
                    account.equity = equity
                account.updated_at = datetime.utcnow()
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error updating account balance: {e}")
            raise
        finally:
            self.close_session(session)
    
    # Asset Management
    def create_asset(self, asset_data: Dict[str, Any]) -> Asset:
        """Create a new asset"""
        session = self.get_session()
        try:
            asset = Asset(**asset_data)
            session.add(asset)
            session.commit()
            session.refresh(asset)
            return asset
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating asset: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get asset by symbol"""
        session = self.get_session()
        try:
            return session.query(Asset).filter(Asset.symbol == symbol).first()
        finally:
            self.close_session(session)
    
    def get_active_assets(self) -> List[Asset]:
        """Get all active assets"""
        session = self.get_session()
        try:
            return session.query(Asset).filter(Asset.is_active == True).all()
        finally:
            self.close_session(session)
    
    # Trade Management
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Create a new trade"""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating trade: {e}")
            raise
        finally:
            self.close_session(session)
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        """Update an existing trade"""
        session = self.get_session()
        try:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if trade:
                for key, value in updates.items():
                    setattr(trade, key, value)
                trade.updated_at = datetime.utcnow()
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error updating trade: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        session = self.get_session()
        try:
            return session.query(Trade).filter(Trade.trade_id == trade_id).first()
        finally:
            self.close_session(session)
    
    def get_trades(self, account_id: str = None, symbol: str = None, 
                   status: str = None, limit: int = 100) -> List[Trade]:
        """Get trades with optional filters"""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if account_id:
                query = query.join(TradingAccount).filter(
                    TradingAccount.account_id == account_id
                )
            
            if symbol:
                query = query.join(Asset).filter(Asset.symbol == symbol)
            
            if status:
                query = query.filter(Trade.status == status)
            
            return query.order_by(desc(Trade.open_time)).limit(limit).all()
        finally:
            self.close_session(session)
    
    def get_trade_statistics(self, account_id: str = None, 
                           start_date: datetime = None, 
                           end_date: datetime = None) -> Dict[str, Any]:
        """Get trading statistics"""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if account_id:
                query = query.join(TradingAccount).filter(
                    TradingAccount.account_id == account_id
                )
            
            if start_date:
                query = query.filter(Trade.open_time >= start_date)
            
            if end_date:
                query = query.filter(Trade.open_time <= end_date)
            
            trades = query.all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_profit_loss': 0.0,
                    'average_profit_loss': 0.0
                }
            
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.status == 'WIN'])
            losing_trades = len([t for t in trades if t.status == 'LOSS'])
            total_profit_loss = sum([t.profit_loss for t in trades])
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0,
                'total_profit_loss': total_profit_loss,
                'average_profit_loss': total_profit_loss / total_trades if total_trades > 0 else 0.0
            }
        finally:
            self.close_session(session)
    
    # Market Data Management
    def save_market_data(self, market_data: List[Dict[str, Any]]):
        """Save market data in batch"""
        session = self.get_session()
        try:
            for data in market_data:
                # Convert indicators dict to JSON string if present
                if 'indicators' in data and isinstance(data['indicators'], dict):
                    data['indicators'] = json.dumps(data['indicators'])
                
                # Check if data already exists
                existing = session.query(MarketData).filter(
                    and_(
                        MarketData.asset_id == data['asset_id'],
                        MarketData.timestamp == data['timestamp'],
                        MarketData.timeframe == data['timeframe']
                    )
                ).first()
                
                if not existing:
                    session.add(MarketData(**data))
            
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error saving market data: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: datetime = None, 
                       end_date: datetime = None, 
                       limit: int = 1000) -> List[MarketData]:
        """Get market data for a symbol"""
        session = self.get_session()
        try:
            query = session.query(MarketData).join(Asset).filter(
                and_(
                    Asset.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            )
            
            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)
            
            return query.order_by(desc(MarketData.timestamp)).limit(limit).all()
        finally:
            self.close_session(session)
    
    def get_latest_market_data(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Get latest market data for a symbol"""
        session = self.get_session()
        try:
            return session.query(MarketData).join(Asset).filter(
                and_(
                    Asset.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            ).order_by(desc(MarketData.timestamp)).first()
        finally:
            self.close_session(session)
    
    # Signal Management
    def create_signal(self, signal_data: Dict[str, Any]) -> TradingSignal:
        """Create a new trading signal"""
        session = self.get_session()
        try:
            # Convert indicators dict to JSON string if present
            if 'indicators_used' in signal_data and isinstance(signal_data['indicators_used'], dict):
                signal_data['indicators_used'] = json.dumps(signal_data['indicators_used'])
            
            signal = TradingSignal(**signal_data)
            session.add(signal)
            session.commit()
            session.refresh(signal)
            return signal
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating signal: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_active_signals(self, account_id: str = None) -> List[TradingSignal]:
        """Get active (not executed) signals"""
        session = self.get_session()
        try:
            query = session.query(TradingSignal).filter(
                TradingSignal.is_executed == False
            )
            
            if account_id:
                query = query.join(TradingAccount).filter(
                    TradingAccount.account_id == account_id
                )
            
            return query.order_by(desc(TradingSignal.generated_at)).all()
        finally:
            self.close_session(session)
    
    def mark_signal_executed(self, signal_id: int, trade_id: str = None):
        """Mark signal as executed"""
        session = self.get_session()
        try:
            signal = session.query(TradingSignal).filter(
                TradingSignal.id == signal_id
            ).first()
            
            if signal:
                signal.is_executed = True
                signal.executed_at = datetime.utcnow()
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error marking signal as executed: {e}")
            raise
        finally:
            self.close_session(session)
    
    # Strategy Management
    def create_strategy(self, strategy_data: Dict[str, Any]) -> Strategy:
        """Create a new strategy"""
        session = self.get_session()
        try:
            # Convert parameters dict to JSON string if present
            if 'parameters' in strategy_data and isinstance(strategy_data['parameters'], dict):
                strategy_data['parameters'] = json.dumps(strategy_data['parameters'])
            
            strategy = Strategy(**strategy_data)
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            return strategy
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating strategy: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_active_strategies(self) -> List[Strategy]:
        """Get active strategies"""
        session = self.get_session()
        try:
            return session.query(Strategy).filter(Strategy.is_active == True).all()
        finally:
            self.close_session(session)
    
    def update_strategy_performance(self, strategy_name: str, 
                                  trade_result: str, profit_loss: float):
        """Update strategy performance metrics"""
        session = self.get_session()
        try:
            strategy = session.query(Strategy).filter(
                Strategy.name == strategy_name
            ).first()
            
            if strategy:
                strategy.total_trades += 1
                
                if trade_result == 'WIN':
                    strategy.winning_trades += 1
                elif trade_result == 'LOSS':
                    strategy.losing_trades += 1
                
                strategy.win_rate = (strategy.winning_trades / strategy.total_trades) * 100
                strategy.profit_loss += profit_loss
                strategy.updated_at = datetime.utcnow()
                
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error updating strategy performance: {e}")
            raise
        finally:
            self.close_session(session)
    
    # ML Model Management
    def create_ml_model(self, model_data: Dict[str, Any]) -> MLModel:
        """Create a new ML model"""
        session = self.get_session()
        try:
            # Convert parameters and features to JSON strings if present
            if 'parameters' in model_data and isinstance(model_data['parameters'], dict):
                model_data['parameters'] = json.dumps(model_data['parameters'])
            
            if 'features' in model_data and isinstance(model_data['features'], list):
                model_data['features'] = json.dumps(model_data['features'])
            
            model = MLModel(**model_data)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating ML model: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_active_ml_models(self) -> List[MLModel]:
        """Get active ML models"""
        session = self.get_session()
        try:
            return session.query(MLModel).filter(MLModel.is_active == True).all()
        finally:
            self.close_session(session)
    
    def save_model_prediction(self, prediction_data: Dict[str, Any]) -> ModelPrediction:
        """Save ML model prediction"""
        session = self.get_session()
        try:
            # Convert features dict to JSON string if present
            if 'features' in prediction_data and isinstance(prediction_data['features'], dict):
                prediction_data['features'] = json.dumps(prediction_data['features'])
            
            prediction = ModelPrediction(**prediction_data)
            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            return prediction
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error saving model prediction: {e}")
            raise
        finally:
            self.close_session(session)
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: str):
        """Update prediction with actual outcome"""
        session = self.get_session()
        try:
            prediction = session.query(ModelPrediction).filter(
                ModelPrediction.id == prediction_id
            ).first()
            
            if prediction:
                prediction.actual_outcome = actual_outcome
                prediction.is_correct = (prediction.prediction_type == actual_outcome)
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error updating prediction outcome: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_model_accuracy(self, model_id: int, days_back: int = 30) -> float:
        """Calculate model accuracy over specified period"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            predictions = session.query(ModelPrediction).filter(
                and_(
                    ModelPrediction.model_id == model_id,
                    ModelPrediction.prediction_time >= cutoff_date,
                    ModelPrediction.actual_outcome.isnot(None)
                )
            ).all()
            
            if not predictions:
                return 0.0
            
            correct_predictions = len([p for p in predictions if p.is_correct])
            return (correct_predictions / len(predictions)) * 100
        finally:
            self.close_session(session)
    
    # Backtest Management
    def save_backtest_result(self, backtest_data: Dict[str, Any]) -> BacktestResult:
        """Save backtest result"""
        session = self.get_session()
        try:
            # Convert lists/dicts to JSON strings if present
            if 'asset_symbols' in backtest_data and isinstance(backtest_data['asset_symbols'], list):
                backtest_data['asset_symbols'] = json.dumps(backtest_data['asset_symbols'])
            
            if 'trade_details' in backtest_data and isinstance(backtest_data['trade_details'], list):
                backtest_data['trade_details'] = json.dumps(backtest_data['trade_details'])
            
            if 'equity_curve' in backtest_data and isinstance(backtest_data['equity_curve'], list):
                backtest_data['equity_curve'] = json.dumps(backtest_data['equity_curve'])
            
            result = BacktestResult(**backtest_data)
            session.add(result)
            session.commit()
            session.refresh(result)
            return result
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error saving backtest result: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_backtest_results(self, strategy_name: str = None) -> List[BacktestResult]:
        """Get backtest results"""
        session = self.get_session()
        try:
            query = session.query(BacktestResult)
            
            if strategy_name:
                query = query.join(Strategy).filter(Strategy.name == strategy_name)
            
            return query.order_by(desc(BacktestResult.created_at)).all()
        finally:
            self.close_session(session)
    
    # Logging
    def log_event(self, level: str, module: str, message: str, **kwargs):
        """Log system event"""
        session = self.get_session()
        try:
            log_entry = SystemLog(
                level=level,
                module=module,
                message=message,
                **kwargs
            )
            session.add(log_entry)
            session.commit()
        except SQLAlchemyError as e:
            # Don't raise exception for logging errors to avoid infinite loops
            print(f"Error logging event: {e}")
        finally:
            self.close_session(session)
    
    def get_logs(self, level: str = None, module: str = None, 
                 start_date: datetime = None, limit: int = 100) -> List[SystemLog]:
        """Get system logs"""
        session = self.get_session()
        try:
            query = session.query(SystemLog)
            
            if level:
                query = query.filter(SystemLog.level == level)
            
            if module:
                query = query.filter(SystemLog.module == module)
            
            if start_date:
                query = query.filter(SystemLog.timestamp >= start_date)
            
            return query.order_by(desc(SystemLog.timestamp)).limit(limit).all()
        finally:
            self.close_session(session)
    
    # Configuration Management
    def set_config(self, key: str, value: Any, description: str = None, 
                   config_type: str = 'general'):
        """Set configuration value"""
        session = self.get_session()
        try:
            # Convert value to string if it's not already
            if not isinstance(value, str):
                value = json.dumps(value)
            
            config = session.query(SystemConfig).filter(
                SystemConfig.key == key
            ).first()
            
            if config:
                config.value = value
                config.updated_at = datetime.utcnow()
                if description:
                    config.description = description
            else:
                config = SystemConfig(
                    key=key,
                    value=value,
                    description=description,
                    config_type=config_type
                )
                session.add(config)
            
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error setting config: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        session = self.get_session()
        try:
            config = session.query(SystemConfig).filter(
                SystemConfig.key == key
            ).first()
            
            if config:
                try:
                    # Try to parse as JSON first
                    return json.loads(config.value)
                except (json.JSONDecodeError, ValueError):
                    # Return as string if not valid JSON
                    return config.value
            
            return default
        finally:
            self.close_session(session)
    
    def get_all_configs(self, config_type: str = None) -> Dict[str, Any]:
        """Get all configuration values"""
        session = self.get_session()
        try:
            query = session.query(SystemConfig)
            
            if config_type:
                query = query.filter(SystemConfig.config_type == config_type)
            
            configs = query.all()
            result = {}
            
            for config in configs:
                try:
                    result[config.key] = json.loads(config.value)
                except (json.JSONDecodeError, ValueError):
                    result[config.key] = config.value
            
            return result
        finally:
            self.close_session(session)
    
    # Data Export/Import
    def export_trades_to_csv(self, filename: str, account_id: str = None, 
                            start_date: datetime = None, end_date: datetime = None):
        """Export trades to CSV file"""
        session = self.get_session()
        try:
            query = session.query(Trade).join(Asset).join(TradingAccount)
            
            if account_id:
                query = query.filter(TradingAccount.account_id == account_id)
            
            if start_date:
                query = query.filter(Trade.open_time >= start_date)
            
            if end_date:
                query = query.filter(Trade.open_time <= end_date)
            
            trades = query.all()
            
            # Convert to DataFrame
            data = []
            for trade in trades:
                data.append({
                    'trade_id': trade.trade_id,
                    'account_id': trade.account.account_id,
                    'symbol': trade.asset.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'amount': trade.amount,
                    'open_time': trade.open_time,
                    'close_time': trade.close_time,
                    'expiry_time': trade.expiry_time,
                    'status': trade.status,
                    'profit_loss': trade.profit_loss,
                    'strategy_name': trade.strategy_name
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Exported {len(trades)} trades to {filename}")
        finally:
            self.close_session(session)
    
    def get_market_data_df(self, symbol: str, timeframe: str, 
                          start_date: datetime = None, 
                          end_date: datetime = None) -> pd.DataFrame:
        """Get market data as pandas DataFrame"""
        session = self.get_session()
        try:
            query = session.query(MarketData).join(Asset).filter(
                and_(
                    Asset.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            )
            
            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)
            
            data = query.order_by(MarketData.timestamp).all()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for candle in data:
                row = {
                    'timestamp': candle.timestamp,
                    'open': candle.open_price,
                    'high': candle.high_price,
                    'low': candle.low_price,
                    'close': candle.close_price,
                    'volume': candle.volume
                }
                
                # Add indicators if available
                if candle.indicators:
                    try:
                        indicators = json.loads(candle.indicators)
                        row.update(indicators)
                    except json.JSONDecodeError:
                        pass
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df
        finally:
            self.close_session(session)
    
    # Cleanup and Maintenance
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old logs
            old_logs = session.query(SystemLog).filter(
                SystemLog.timestamp < cutoff_date
            ).delete()
            
            # Clean up old market data (keep only daily and higher timeframes)
            old_market_data = session.query(MarketData).filter(
                and_(
                    MarketData.timestamp < cutoff_date,
                    MarketData.timeframe.in_(['1m', '5m', '15m', '30m'])
                )
            ).delete()
            
            # Clean up old predictions
            old_predictions = session.query(ModelPrediction).filter(
                ModelPrediction.prediction_time < cutoff_date
            ).delete()
            
            session.commit()
            
            self.logger.info(f"Cleaned up {old_logs} logs, {old_market_data} market data records, "
                           f"and {old_predictions} predictions older than {days_to_keep} days")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
        finally:
            self.close_session(session)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        session = self.get_session()
        try:
            stats = {}
            
            # Count records in each table
            stats['accounts'] = session.query(TradingAccount).count()
            stats['assets'] = session.query(Asset).count()
            stats['trades'] = session.query(Trade).count()
            stats['market_data'] = session.query(MarketData).count()
            stats['signals'] = session.query(TradingSignal).count()
            stats['strategies'] = session.query(Strategy).count()
            stats['ml_models'] = session.query(MLModel).count()
            stats['predictions'] = session.query(ModelPrediction).count()
            stats['backtest_results'] = session.query(BacktestResult).count()
            stats['logs'] = session.query(SystemLog).count()
            stats['configs'] = session.query(SystemConfig).count()
            
            return stats
        finally:
            self.close_session(session)
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.engine.dispose()
        except:
            pass