"""
Prediction Engine for Quotex Trading Bot
Advanced ML-based price prediction system with ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Local imports
from ..utils.logger import setup_logger
from ..utils.helpers import validate_data, calculate_metrics
from .feature_engineering import FeatureEngineer


class PredictionEngine:
    """
    Advanced prediction engine using ensemble ML models for price forecasting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prediction engine
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.feature_engineer = FeatureEngineer()
        
        # Model parameters
        self.lookback_window = config.get('lookback_window', 60)
        self.prediction_horizon = config.get('prediction_horizon', 5)
        self.model_type = config.get('model_type', 'ensemble')
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.model_weights = {}
        self.is_trained = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.prediction_history = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with optimized parameters"""
        try:
            # Random Forest
            self.models['rf'] = RandomForestRegressor(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', 10),
                min_samples_split=self.config.get('rf_min_samples_split', 5),
                min_samples_leaf=self.config.get('rf_min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=self.config.get('gb_n_estimators', 100),
                learning_rate=self.config.get('gb_learning_rate', 0.1),
                max_depth=self.config.get('gb_max_depth', 6),
                random_state=42
            )
            
            # Support Vector Regression
            self.models['svr'] = SVR(
                kernel=self.config.get('svr_kernel', 'rbf'),
                C=self.config.get('svr_C', 1.0),
                gamma=self.config.get('svr_gamma', 'scale')
            )
            
            # Linear models
            self.models['linear'] = LinearRegression()
            self.models['ridge'] = Ridge(
                alpha=self.config.get('ridge_alpha', 1.0),
                random_state=42
            )
            
            # Initialize scalers
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            # Set initial weights (will be updated during training)
            num_models = len(self.models)
            for model_name in self.models.keys():
                self.model_weights[model_name] = 1.0 / num_models
            
            self.logger.info(f"Initialized {len(self.models)} models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def prepare_training_data(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and targets
        
        Args:
            market_data: Historical market data
            
        Returns:
            Tuple of (features, targets)
        """
        try:
            # Validate input data
            if not validate_data(market_data):
                raise ValueError("Invalid market data provided")
            
            # Generate features
            features_df = self.feature_engineer.generate_features(market_data)
            
            # Create target variable (future price movement)
            targets = self._create_targets(market_data)
            
            # Align features and targets
            min_length = min(len(features_df), len(targets))
            features = features_df.iloc[:min_length].values
            targets = targets[:min_length]
            
            # Remove NaN values
            valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
            features = features[valid_indices]
            targets = targets[valid_indices]
            
            self.logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
            
            return features, targets
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _create_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Create target variables for prediction
        
        Args:
            market_data: Historical market data
            
        Returns:
            Target array
        """
        try:
            prices = market_data['close'].values
            targets = np.zeros(len(prices))
            
            # Calculate future price movement
            for i in range(len(prices) - self.prediction_horizon):
                future_price = prices[i + self.prediction_horizon]
                current_price = prices[i]
                
                # Calculate percentage change
                targets[i] = (future_price - current_price) / current_price * 100
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error creating targets: {str(e)}")
            raise
    
    def train_models(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train all models and calculate performance metrics
        
        Args:
            market_data: Historical market data for training
            
        Returns:
            Dictionary of model performance scores
        """
        try:
            # Prepare training data
            features, targets = self.prepare_training_data(market_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            model_scores = {}
            
            # Train each model
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name} model...")
                
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'
                )
                cv_score = -cv_scores.mean()
                
                model_scores[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_score': cv_score
                }
                
                self.logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Update model weights based on performance
            self._update_model_weights(model_scores)
            
            # Store performance metrics
            self.performance_metrics = model_scores
            self.is_trained = True
            
            self.logger.info("All models trained successfully")
            
            return model_scores
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
    
    def _update_model_weights(self, model_scores: Dict[str, Dict[str, float]]):
        """
        Update model weights based on performance scores
        
        Args:
            model_scores: Dictionary of model performance metrics
        """
        try:
            # Use inverse of MSE for weights (lower MSE = higher weight)
            total_inverse_mse = 0
            inverse_mse_scores = {}
            
            for model_name, scores in model_scores.items():
                inverse_mse = 1.0 / (scores['mse'] + 1e-8)  # Add small epsilon to avoid division by zero
                inverse_mse_scores[model_name] = inverse_mse
                total_inverse_mse += inverse_mse
            
            # Normalize weights
            for model_name in inverse_mse_scores:
                self.model_weights[model_name] = inverse_mse_scores[model_name] / total_inverse_mse
            
            self.logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating model weights: {str(e)}")
            raise
    
    def predict(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make price predictions using ensemble of models
        
        Args:
            market_data: Recent market data for prediction
            
        Returns:
            Dictionary containing predictions and confidence metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before making predictions")
            
            # Generate features for latest data
            features_df = self.feature_engineer.generate_features(market_data)
            
            # Get the latest feature vector
            latest_features = features_df.iloc[-1].values.reshape(1, -1)
            
            # Check for NaN values
            if np.isnan(latest_features).any():
                self.logger.warning("NaN values detected in features, using last valid values")
                latest_features = np.nan_to_num(latest_features)
            
            predictions = {}
            scaled_predictions = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                # Scale features
                scaled_features = self.scalers[model_name].transform(latest_features)
                
                # Make prediction
                pred = model.predict(scaled_features)[0]
                predictions[model_name] = pred
                
                # Weight the prediction
                weighted_pred = pred * self.model_weights[model_name]
                scaled_predictions.append(weighted_pred)
            
            # Ensemble prediction
            ensemble_prediction = sum(scaled_predictions)
            
            # Calculate prediction confidence
            pred_values = list(predictions.values())
            prediction_std = np.std(pred_values)
            confidence = max(0, 1 - (prediction_std / (abs(ensemble_prediction) + 1e-8)))
            
            # Determine signal direction
            signal = self._generate_signal(ensemble_prediction, confidence)
            
            # Create prediction result
            result = {
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'confidence': confidence,
                'signal': signal,
                'timestamp': datetime.now(),
                'model_weights': self.model_weights.copy()
            }
            
            # Store prediction history
            self.prediction_history.append(result)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            self.logger.info(f"Prediction: {ensemble_prediction:.4f}, Confidence: {confidence:.4f}, Signal: {signal}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _generate_signal(self, prediction: float, confidence: float) -> str:
        """
        Generate trading signal based on prediction and confidence
        
        Args:
            prediction: Ensemble prediction value
            confidence: Prediction confidence score
            
        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD')
        """
        try:
            min_confidence = self.config.get('min_confidence', 0.6)
            min_prediction_threshold = self.config.get('min_prediction_threshold', 0.1)
            
            # Check if confidence is sufficient
            if confidence < min_confidence:
                return 'HOLD'
            
            # Generate signal based on prediction magnitude
            if prediction > min_prediction_threshold:
                return 'BUY'
            elif prediction < -min_prediction_threshold:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return 'HOLD'
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get comprehensive model performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            if not self.performance_metrics:
                return {"error": "No performance metrics available. Train models first."}
            
            # Calculate additional metrics
            performance_summary = {
                'individual_models': self.performance_metrics,
                'model_weights': self.model_weights,
                'is_trained': self.is_trained,
                'prediction_history_length': len(self.prediction_history)
            }
            
            # Calculate recent prediction accuracy if available
            if len(self.prediction_history) > 10:
                recent_predictions = self.prediction_history[-10:]
                recent_confidence = np.mean([p['confidence'] for p in recent_predictions])
                performance_summary['recent_avg_confidence'] = recent_confidence
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {str(e)}")
            return {"error": str(e)}
    
    def save_models(self, filepath: str):
        """
        Save trained models to disk
        
        Args:
            filepath: Path to save models
        """
        try:
            if not self.is_trained:
                raise ValueError("No trained models to save")
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'model_weights': self.model_weights,
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath: str):
        """
        Load trained models from disk
        
        Args:
            filepath: Path to load models from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_weights = model_data['model_weights']
            self.performance_metrics = model_data['performance_metrics']
            self.config.update(model_data['config'])
            
            self.is_trained = True
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def retrain_models(self, market_data: pd.DataFrame):
        """
        Retrain models with new data
        
        Args:
            market_data: New market data for retraining
        """
        try:
            self.logger.info("Retraining models with new data...")
            
            # Reset models
            self._initialize_models()
            
            # Train with new data
            self.train_models(market_data)
            
            self.logger.info("Models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
            raise
    
    def optimize_parameters(self, market_data: pd.DataFrame, param_ranges: Dict[str, Any]):
        """
        Optimize model parameters using grid search
        
        Args:
            market_data: Training data
            param_ranges: Dictionary of parameter ranges to search
        """
        try:
            from sklearn.model_selection import GridSearchCV
            
            self.logger.info("Starting parameter optimization...")
            
            # Prepare data
            features, targets = self.prepare_training_data(market_data)
            
            best_params = {}
            
            # Optimize each model
            for model_name, model in self.models.items():
                if model_name in param_ranges:
                    self.logger.info(f"Optimizing {model_name} parameters...")
                    
                    # Scale features
                    scaled_features = self.scalers[model_name].fit_transform(features)
                    
                    # Grid search
                    grid_search = GridSearchCV(
                        model, param_ranges[model_name], 
                        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                    )
                    grid_search.fit(scaled_features, targets)
                    
                    # Update model with best parameters
                    best_params[model_name] = grid_search.best_params_
                    self.models[model_name] = grid_search.best_estimator_
                    
                    self.logger.info(f"{model_name} best params: {grid_search.best_params_}")
            
            # Update config with best parameters
            self.config['optimized_params'] = best_params
            
            self.logger.info("Parameter optimization completed")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models
        
        Returns:
            Dictionary of feature importance arrays
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained first")
            
            importance_dict = {}
            
            # Get feature importance from tree-based models
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_dict[model_name] = model.feature_importances_
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def validate_prediction_accuracy(self, market_data: pd.DataFrame, 
                                   validation_days: int = 30) -> Dict[str, float]:
        """
        Validate prediction accuracy on recent data
        
        Args:
            market_data: Recent market data for validation
            validation_days: Number of days to validate
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained first")
            
            # Get recent data for validation
            recent_data = market_data.tail(validation_days * 24)  # Assuming hourly data
            
            actual_movements = []
            predicted_movements = []
            
            # Generate predictions for each time step
            for i in range(len(recent_data) - self.prediction_horizon):
                # Get data up to current point
                current_data = recent_data.iloc[:i+self.lookback_window]
                
                if len(current_data) < self.lookback_window:
                    continue
                
                # Make prediction
                try:
                    prediction_result = self.predict(current_data)
                    predicted_movement = prediction_result['ensemble_prediction']
                    
                    # Calculate actual movement
                    current_price = recent_data.iloc[i]['close']
                    future_price = recent_data.iloc[i + self.prediction_horizon]['close']
                    actual_movement = (future_price - current_price) / current_price * 100
                    
                    actual_movements.append(actual_movement)
                    predicted_movements.append(predicted_movement)
                    
                except Exception as e:
                    self.logger.warning(f"Error in validation step {i}: {str(e)}")
                    continue
            
            if len(actual_movements) < 10:
                return {"error": "Insufficient data for validation"}
            
            # Calculate accuracy metrics
            actual_movements = np.array(actual_movements)
            predicted_movements = np.array(predicted_movements)
            
            # Direction accuracy
            actual_directions = np.sign(actual_movements)
            predicted_directions = np.sign(predicted_movements)
            direction_accuracy = np.mean(actual_directions == predicted_directions)
            
            # Magnitude accuracy
            mse = mean_squared_error(actual_movements, predicted_movements)
            mae = mean_absolute_error(actual_movements, predicted_movements)
            
            # Correlation
            correlation = np.corrcoef(actual_movements, predicted_movements)[0, 1]
            
            validation_metrics = {
                'direction_accuracy': direction_accuracy,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'sample_size': len(actual_movements)
            }
            
            self.logger.info(f"Validation metrics: {validation_metrics}")
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating prediction accuracy: {str(e)}")
            return {"error": str(e)}