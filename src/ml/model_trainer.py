"""
Model Trainer for Quotex Trading Bot
Handles training of various ML models for price prediction and signal generation
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import setup_logger
from config import settings
from .feature_engineering import FeatureEngineer


class ModelTrainer:
    """
    Comprehensive model training class for trading predictions
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize the model trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        self.logger = setup_logger(__name__)
        self.feature_engineer = FeatureEngineer()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models dictionary
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        }
        
        # Model performance tracking
        self.model_performance = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'signal') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            data: Input DataFrame with features and target
            target_column: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        try:
            # Generate features using feature engineering
            features_df = self.feature_engineer.generate_features(data)
            
            # Handle missing values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare target variable
            if target_column not in features_df.columns:
                # Generate trading signals based on price movement
                features_df['signal'] = self._generate_signals(features_df)
                target_column = 'signal'
            
            # Separate features and target
            X = features_df.drop(columns=[target_column])
            y = features_df[target_column]
            
            # Handle categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            self.logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            return X_scaled, y.values
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals based on price movement
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Array of trading signals (0: sell, 1: buy, 2: hold)
        """
        try:
            # Calculate future returns
            future_return = data['close'].pct_change().shift(-1)
            
            # Define signal thresholds
            buy_threshold = 0.001  # 0.1% positive return
            sell_threshold = -0.001  # -0.1% negative return
            
            # Generate signals
            signals = np.where(future_return > buy_threshold, 1,
                             np.where(future_return < sell_threshold, 0, 2))
            
            return signals[:-1]  # Remove last element (no future data)
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return np.zeros(len(data) - 1)
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with model performance metrics
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Train the model
            self.logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Calculate metrics
            y_pred = model.predict(X_train)
            metrics = {
                'accuracy': accuracy_score(y_train, y_pred),
                'precision': precision_score(y_train, y_pred, average='weighted'),
                'recall': recall_score(y_train, y_pred, average='weighted'),
                'f1_score': f1_score(y_train, y_pred, average='weighted'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.logger.info(f"{model_name} training completed - Accuracy: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            return {}
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a model
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best parameters and performance
        """
        try:
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
            
            if model_name not in param_grids:
                self.logger.warning(f"No parameter grid defined for {model_name}")
                return {}
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.models[model_name],
                param_grids[model_name],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            self.logger.info(f"Starting hyperparameter tuning for {model_name}...")
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return {}
    
    def train_all_models(self, data: pd.DataFrame, target_column: str = 'signal', 
                        test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            data: Training data
            target_column: Target column name
            test_size: Test set size ratio
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with all model performances
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            results = {}
            
            for model_name in self.models.keys():
                self.logger.info(f"Processing {model_name}...")
                
                # Hyperparameter tuning
                if tune_hyperparameters:
                    tuning_results = self.hyperparameter_tuning(model_name, X_train, y_train)
                    results[f"{model_name}_tuning"] = tuning_results
                
                # Train model
                train_metrics = self.train_model(model_name, X_train, y_train)
                
                # Test model
                test_metrics = self.evaluate_model(model_name, X_test, y_test)
                
                # Combine results
                results[model_name] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'model_object': self.models[model_name]
                }
            
            # Find best model
            self.best_model = self._find_best_model(results)
            self.model_performance = results
            
            self.logger.info(f"Best model: {self.best_model}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training all models: {str(e)}")
            return {}
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {}
    
    def _find_best_model(self, results: Dict[str, Any]) -> str:
        """
        Find the best performing model
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Name of the best model
        """
        best_model = None
        best_score = -1
        
        for model_name, model_results in results.items():
            if 'test_metrics' in model_results:
                score = model_results['test_metrics'].get('f1_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    def save_model(self, model_name: str, filename: Optional[str] = None) -> bool:
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            filename: Optional filename (default: model_name.pkl)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            if filename is None:
                filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            filepath = os.path.join(self.model_dir, filename)
            
            # Save model and scaler
            model_data = {
                'model': self.models[model_name],
                'scaler': self.scaler,
                'feature_engineer': self.feature_engineer,
                'performance': self.model_performance.get(model_name, {}),
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model {model_name} saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model
        
        Args:
            filepath: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            
            # Extract components
            model_name = os.path.basename(filepath).split('_')[0]
            self.models[model_name] = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_engineer = model_data['feature_engineer']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from {filepath}: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            data: Input data for prediction
            model_name: Name of the model to use (default: best model)
            
        Returns:
            Array of predictions
        """
        try:
            if model_name is None:
                model_name = self.best_model
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Prepare features
            features_df = self.feature_engineer.generate_features(data)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Handle categorical variables
            for col in features_df.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                features_df[col] = le.fit_transform(features_df[col].astype(str))
            
            # Scale features
            X_scaled = self.scaler.transform(features_df)
            
            # Make predictions
            predictions = self.models[model_name].predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance from a trained model
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        try:
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                self.logger.warning(f"Model {model_name} doesn't support feature importance")
                return pd.DataFrame()
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'total_models': len(self.models),
            'best_model': self.best_model,
            'model_performance': self.model_performance,
            'training_timestamp': datetime.now().isoformat()
        }
        
        return summary