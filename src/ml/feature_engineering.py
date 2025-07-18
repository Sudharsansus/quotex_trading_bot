"""
Model Trainer for Quotex Trading Bot
Handles training, validation, and management of ML models for trading predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json

from ..utils.logger import setup_logger
from .feature_engineering import FeatureEngineer
from ..database.database_manager import DatabaseManager


class ModelTrainer:
    """
    Advanced model trainer for trading predictions with multiple ML algorithms
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelTrainer
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.feature_engineer = FeatureEngineer(config)
        self.db_manager = DatabaseManager(config)
        
        # Model storage paths
        self.model_dir = config.get('model_dir', 'data/models')
        self.scaler_dir = config.get('scaler_dir', 'data/scalers')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # Initialize models
        self.models = self._initialize_models()
        self.scalers = {}
        self.trained_models = {}
        self.model_metrics = {}
        
        # Training parameters
        self.test_size = config.get('test_size', 0.2)
        self.validation_size = config.get('validation_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.cv_folds = config.get('cv_folds', 5)
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models with default parameters"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
    
    def prepare_training_data(self, 
                            symbol: str, 
                            timeframe: str, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and labels
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        try:
            # Get historical data
            historical_data = self.db_manager.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if historical_data.empty:
                raise ValueError(f"No historical data found for {symbol} {timeframe}")
            
            # Generate features
            features_df = self.feature_engineer.generate_features(historical_data)
            
            # Generate labels (1 for up, 0 for down)
            labels = self._generate_labels(historical_data)
            
            # Remove NaN values
            combined_df = features_df.join(labels, how='inner')
            combined_df = combined_df.dropna()
            
            features_df = combined_df.iloc[:, :-1]
            labels_series = combined_df.iloc[:, -1]
            
            self.logger.info(f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
            
            return features_df, labels_series
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _generate_labels(self, data: pd.DataFrame, 
                        lookforward: int = 1) -> pd.Series:
        """
        Generate trading labels based on future price movement
        
        Args:
            data: Historical price data
            lookforward: Number of periods to look forward
            
        Returns:
            Series of labels (1 for up, 0 for down)
        """
        try:
            # Calculate future returns
            future_returns = data['close'].pct_change(lookforward).shift(-lookforward)
            
            # Generate binary labels
            labels = (future_returns > 0).astype(int)
            labels.name = 'label'
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error generating labels: {str(e)}")
            raise
    
    def train_model(self, 
                   model_name: str,
                   features: pd.DataFrame,
                   labels: pd.Series,
                   optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            features: Feature matrix
            labels: Target labels
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary containing training results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[model_name] = scaler
            
            # Get model
            model = self.models[model_name]
            
            # Hyperparameter optimization
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(model_name, X_train_scaled, y_train)
            
            # Train model
            self.logger.info(f"Training {model_name} model...")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.cv_folds)
            
            # Store trained model
            self.trained_models[model_name] = model
            
            # Prepare results
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'feature_importance': self._get_feature_importance(model, features.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store metrics
            self.model_metrics[model_name] = results
            
            self.logger.info(f"Model {model_name} trained successfully. Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {str(e)}")
            raise
    
    def _optimize_hyperparameters(self, 
                                 model_name: str, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray) -> Any:
        """
        Optimize hyperparameters using GridSearchCV
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Optimized model
        """
        try:
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'min_samples_split': [2, 5, 10]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                },
                'svm': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
            
            if model_name not in param_grids:
                return self.models[model_name]
            
            self.logger.info(f"Optimizing hyperparameters for {model_name}...")
            
            grid_search = GridSearchCV(
                self.models[model_name],
                param_grids[model_name],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters for {model_name}: {str(e)}")
            return self.models[model_name]
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                for name, importance in zip(feature_names, importances):
                    importance_dict[name] = float(importance)
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def train_all_models(self, 
                        features: pd.DataFrame,
                        labels: pd.Series,
                        optimize_hyperparameters: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models
        
        Args:
            features: Feature matrix
            labels: Target labels
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of all training results
        """
        try:
            all_results = {}
            
            for model_name in self.models.keys():
                self.logger.info(f"Training {model_name}...")
                results = self.train_model(
                    model_name, 
                    features, 
                    labels, 
                    optimize_hyperparameters
                )
                all_results[model_name] = results
            
            # Find best model
            best_model = max(all_results.keys(), key=lambda x: all_results[x]['accuracy'])
            self.logger.info(f"Best performing model: {best_model} with accuracy: {all_results[best_model]['accuracy']:.4f}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error training all models: {str(e)}")
            raise
    
    def save_model(self, model_name: str, symbol: str, timeframe: str) -> str:
        """
        Save trained model and scaler to disk
        
        Args:
            model_name: Name of the model to save
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Path to saved model
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained yet")
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{symbol}_{timeframe}_{timestamp}.pkl"
            scaler_filename = f"{model_name}_{symbol}_{timeframe}_{timestamp}_scaler.pkl"
            
            # Save model
            model_path = os.path.join(self.model_dir, model_filename)
            joblib.dump(self.trained_models[model_name], model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.scaler_dir, scaler_filename)
            joblib.dump(self.scalers[model_name], scaler_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metrics': self.model_metrics.get(model_name, {}),
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_filename = f"{model_name}_{symbol}_{timeframe}_{timestamp}_metadata.json"
            metadata_path = os.path.join(self.model_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_path: str, scaler_path: str) -> Tuple[Any, Any]:
        """
        Load saved model and scaler
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            
        Returns:
            Tuple of (model, scaler)
        """
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            self.logger.info(f"Model loaded from: {model_path}")
            
            return model, scaler
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate_model(self, 
                      model_name: str,
                      test_features: pd.DataFrame,
                      test_labels: pd.Series) -> Dict[str, Any]:
        """
        Evaluate trained model on test data
        
        Args:
            model_name: Name of the model to evaluate
            test_features: Test features
            test_labels: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained yet")
            
            model = self.trained_models[model_name]
            scaler = self.scalers[model_name]
            
            # Scale test features
            test_features_scaled = scaler.transform(test_features)
            
            # Make predictions
            predictions = model.predict(test_features_scaled)
            prediction_probabilities = model.predict_proba(test_features_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            
            evaluation_results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'classification_report': classification_report(test_labels, predictions, output_dict=True),
                'confusion_matrix': confusion_matrix(test_labels, predictions).tolist(),
                'test_samples': len(test_features),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Model {model_name} evaluation completed. Accuracy: {accuracy:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all trained models
        
        Returns:
            DataFrame with model comparison metrics
        """
        try:
            if not self.model_metrics:
                return pd.DataFrame()
            
            comparison_data = []
            for model_name, metrics in self.model_metrics.items():
                comparison_data.append({
                    'model_name': model_name,
                    'accuracy': metrics['accuracy'],
                    'cv_mean': metrics['cv_mean'],
                    'cv_std': metrics['cv_std'],
                    'training_samples': metrics['training_samples'],
                    'test_samples': metrics['test_samples']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_best_model(self) -> Optional[str]:
        """
        Get the name of the best performing model
        
        Returns:
            Name of the best model or None if no models trained
        """
        try:
            if not self.model_metrics:
                return None
            
            best_model = max(self.model_metrics.keys(), 
                           key=lambda x: self.model_metrics[x]['accuracy'])
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error getting best model: {str(e)}")
            return None