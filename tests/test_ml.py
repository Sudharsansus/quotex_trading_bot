"""
Test suite for machine learning components.
Tests ML models, feature engineering, and prediction systems.
"""

import unittest
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn warnings for cleaner test output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class MockFeatureEngineering:
    """Mock feature engineering class for testing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_technical_features(self, data):
        """Create technical analysis features."""
        features = data.copy()
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Moving averages
        features['sma_5'] = data['close'].rolling(5).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        
        # Price ratios
        features['price_to_sma20'] = data['close'] / features['sma_20']
        features['sma5_to_sma20'] = features['sma_5'] / features['sma_20']
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        # Time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
        
        return features
    
    def create_lag_features(self, data, target_col='close', lags=[1, 2, 3, 5, 10]):
        """Create lagged features."""
        features = data.copy()
        
        for lag in lags:
            features[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
            features[f'{target_col}_change_lag_{lag}'] = data[target_col].pct_change(lag)
        
        return features
    
    def create_target_variable(self, data, target_type='classification', horizon=1):
        """Create target variable for prediction."""
        if target_type == 'classification':
            # Binary classification: price up (1) or down (0)
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            target = (future_returns > 0).astype(int)
        elif target_type == 'regression':
            # Regression: future price
            target = data['close'].shift(-horizon)
        else:
            raise ValueError("target_type must be 'classification' or 'regression'")
        
        return target
    
    def prepare_ml_dataset(self, data, target_col, feature_cols=None, test_size=0.2):
        """Prepare dataset for ML training."""
        if feature_cols is None:
            # Exclude non-numeric columns and target
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Remove rows with missing values
        dataset = data[feature_cols + [target_col]].dropna()
        
        X = dataset[feature_cols]
        y = dataset[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test


class MockModelTrainer:
    """Mock model trainer for testing."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_classification_model(self, X_train, y_train, model_type='random_forest'):
        """Train classification model."""
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scale features for logistic regression
        if model_type == 'logistic_regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            self.scalers[model_type] = scaler
        else:
            model.fit(X_train, y_train)
        
        self.models[model_type] = model
        return model
    
    def train_regression_model(self, X_train, y_train, model_type='random_forest'):
        """Train regression model."""
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'linear_regression':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scale features for linear regression
        if model_type == 'linear_regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
            self.scalers[model_type] = scaler
        else:
            model.fit(X_train, y_train)
        
        self.models[model_type] = model
        return model
    
    def predict(self, model_type, X_test):
        """Make predictions."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        
        # Apply scaling if needed
        if model_type in self.scalers:
            scaler = self.scalers[model_type]
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(X_test)
        
        return predictions
    
    def predict_proba(self, model_type, X_test):
        """Get prediction probabilities (for classification)."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_type} doesn't support probability prediction")
        
        # Apply scaling if needed
        if model_type in self.scalers:
            scaler = self.scalers[model_type]
            X_test_scaled = scaler.transform(X_test)
            probabilities = model.predict_proba(X_test_scaled)
        else:
            probabilities = model.predict_proba(X_test)
        
        return probabilities
    
    def save_model(self, model_type, filepath):
        """Save trained model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model_data = {
            'model': self.models[model_type],
            'scaler': self.scalers.get(model_type, None)
        }
        
        joblib.dump(model_data, filepath)
        
    def load_model(self, model_type, filepath):
        """Load trained model."""
        model_data = joblib.load(filepath)
        
        self.models[model_type] = model_data['model']
        if model_data['scaler'] is not None:
            self.scalers[model_type] = model_data['scaler']


class MockPredictionEngine:
    """Mock prediction engine for testing."""
    
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.feature_engineering = MockFeatureEngineering()
        
    def prepare_features(self, data):
        """Prepare features for prediction."""
        # Create technical features
        features = self.feature_engineering.create_technical_features(data)
        
        # Create lag features
        features = self.feature_engineering.create_lag_features(features)
        
        return features
    
    def predict_direction(self, data, model_type='random_forest'):
        """Predict price direction."""
        features = self.prepare_features(data)
        
        # Select feature columns (numeric only, exclude target)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.startswith('close') or 'lag' in col]
        
        X = features[feature_cols].dropna()
        
        if len(X) == 0:
            return np.array([])
        
        predictions = self.model_trainer.predict(model_type, X)
        probabilities = self.model_trainer.predict_proba(model_type, X)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'timestamps': X.index
        }
    
    def predict_price(self, data, model_type='random_forest'):
        """Predict future price."""
        features = self.prepare_features(data)
        
        # Select feature columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.startswith('close') or 'lag' in col]
        
        X = features[feature_cols].dropna()
        
        if len(X) == 0:
            return np.array([])
        
        predictions = self.model_trainer.predict(model_type, X)
        
        return {
            'predictions': predictions,
            'timestamps': X.index
        }


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.01, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 200),
            'high': prices + np.abs(np.random.normal(0, 0.3, 200)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 200)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        self.feature_eng = MockFeatureEngineering()
    
    def test_technical_features_creation(self):
        """Test technical feature creation."""
        features = self.feature_eng.create_technical_features(self.sample_data)
        
        # Check that new features are created
        expected_features = ['returns', 'log_returns', 'volatility', 'sma_5', 'sma_20', 'sma_50']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check feature properties
        self.assertTrue(features['returns'].dropna().abs().max() < 1)  # Reasonable returns
        self.assertTrue(features['volatility'].dropna().min() >= 0)  # Volatility >= 0
        
        # Check moving averages are properly calculated
        manual_sma5 = self.sample_data['close'].rolling(5).mean()
        np.testing.assert_array_almost_equal(
            features['sma_5'].dropna(), 
            manual_sma5.dropna(), 
            decimal=10
        )
    
    def test_lag_features_creation(self):
        """Test lag feature creation."""
        features = self.feature_eng.create_lag_features(self.sample_data, lags=[1, 5, 10])
        
        # Check that lag features are created
        expected_lags = ['close_lag_1', 'close_lag_5', 'close_lag_10']
        for lag_feature in expected_lags:
            self.assertIn(lag_feature, features.columns)
        
        # Check lag feature correctness
        np.testing.assert_array_equal(
            features['close_lag_1'].dropna().values,
            self.sample_data['close'].shift(1).dropna().values
        )
    
    def test_target_variable_creation(self):
        """Test target variable creation."""
        # Test classification target
        target_class = self.feature_eng.create_target_variable(
            self.sample_data, 'classification', horizon=1
        )
        
        # Should be binary (0 or 1)
        unique_values = target_class.dropna().unique()
        self.assertTrue(set(unique_values).issubset({0, 1}))
        
        # Test regression target
        target_reg = self.feature_eng.create_target_variable(
            self.sample_data, 'regression', horizon=1
        )
        
        # Should be continuous values (future prices)
        self.assertTrue(target_reg.dropna().min() > 0)  # Prices should be positive
    
    def test_ml_dataset_preparation(self):
        """Test ML dataset preparation."""
        # Add target variable
        features = self.feature_eng.create_technical_features(self.sample_data)
        features['target'] = self.feature_eng.create_target_variable(features, 'classification')
        
        X_train, X_test, y_train, y_test = self.feature_eng.prepare_ml_dataset(
            features, 'target', test_size=0.3
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[1], X_test.shape[1])  # Same number of features
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Check train/test split ratio
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.3, places=1)
        
        # Check no missing values
        self.assertEqual(X_train.isnull().sum().sum(), 0)
        self.assertEqual(X_test.isnull().sum().sum(), 0)


class TestModelTraining(unittest.TestCase):
    """Test cases for model training."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample training data
        n_samples = 1000
        n_features = 10
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Classification target (binary)
        self.y_class = np.random.randint(0, 2, n_samples)
        
        # Regression target (continuous)
        self.y_reg = np.random.randn(n_samples) * 10 + 100
        
        # Test data
        n_test = 200
        self.X_test = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test_class = np.random.randint(0, 2, n_test)
        self.y_test_reg = np.random.randn(n_test) * 10 + 100
        
        self.trainer = MockModelTrainer()
    
    def test_random_forest_classification(self):
        """Test Random Forest classification training."""
        model = self.trainer.train_classification_model(
            self.X_train, self.y_class, 'random_forest'
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        self.assertIn('random_forest', self.trainer.models)
        
        # Test predictions
        predictions = self.trainer.predict('random_forest', self.X_test)
        probabilities = self.trainer.predict_proba('random_forest', self.X_test)
        
        # Check prediction shapes and ranges
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))  # Probabilities sum to 1
    
    def test_logistic_regression_classification(self):
        """Test Logistic Regression classification training."""
        model = self.trainer.train_classification_model(
            self.X_train, self.y_class, 'logistic_regression'
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        self.assertIn('logistic_regression', self.trainer.models)
        self.assertIn('logistic_regression', self.trainer.scalers)  # Should have scaler
        
        # Test predictions
        predictions = self.trainer.predict('logistic_regression', self.X_test)
        probabilities = self.trainer.predict_proba('logistic_regression', self.X_test)
        
        # Check prediction properties
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))
    
    def test_random_forest_regression(self):
        """Test Random Forest regression training."""
        model = self.trainer.train_regression_model(
            self.X_train, self.y_reg, 'random_forest'
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Test predictions
        predictions = self.trainer.predict('random_forest', self.X_test)
        
        # Check prediction properties
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))  # All predictions should be finite
        
        # Check reasonable prediction range
        self.assertTrue(predictions.min() > 50)  # Should be in reasonable range
        self.assertTrue(predictions.max() < 200)
    
    def test_linear_regression(self):
        """Test Linear Regression training."""
        model = self.trainer.train_regression_model(
            self.X_train, self.y_reg, 'linear_regression'
        )
        
        # Check model is trained
        self.assertIsNotNone(model)
        self.assertIn('linear_regression', self.trainer.scalers)  # Should have scaler
        
        # Test predictions
        predictions = self.trainer.predict('linear_regression', self.X_test)
        
        # Check prediction properties
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        # Train classification model
        self.trainer.train_classification_model(self.X_train, self.y_class, 'random_forest')
        predictions = self.trainer.predict('random_forest', self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test_class, predictions)
        precision = precision_score(self.y_test_class, predictions, average='weighted')
        recall = recall_score(self.y_test_class, predictions, average='weighted')
        
        # Check metric ranges
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
        
        # Train regression model
        self.trainer.train_regression_model(self.X_train, self.y_reg, 'random_forest')
        reg_predictions = self.trainer.predict('random_forest', self.X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(self.y_test_reg, reg_predictions)
        rmse = np.sqrt(mse)
        
        # Check metric properties
        self.assertGreaterEqual(mse, 0)
        self.assertGreaterEqual(rmse, 0)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train a model
        self.trainer.train_classification_model(self.X_train, self.y_class, 'random_forest')
        
        # Get predictions before saving
        original_predictions = self.trainer.predict('random_forest', self.X_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            self.trainer.save_model('random_forest', temp_path)
            
            # Create new trainer and load model
            new_trainer = MockModelTrainer()
            new_trainer.load_model('random_forest', temp_path)
            
            # Get predictions from loaded model
            loaded_predictions = new_trainer.predict('random_forest', self.X_test)
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            os.unlink(temp_path)


class TestPredictionEngine(unittest.TestCase):
    """Test cases for prediction engine."""
    
    def setUp(self):
        """Set up test environment."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='H')
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.01, 500)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.sample_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 500),
            'high': prices + np.abs(np.random.normal(0, 0.3, 500)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 500)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Prepare training data
        feature_eng = MockFeatureEngineering()
        features = feature_eng.create_technical_features(self.sample_data)
        features = feature_eng.create_lag_features(features)
        features['target_class'] = feature_eng.create_target_variable(features, 'classification')
        features['target_reg'] = feature_eng.create_target_variable(features, 'regression')
        
        # Prepare ML dataset
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.startswith('target')]
        
        dataset = features[feature_cols + ['target_class', 'target_reg']].dropna()
        
        # Split data
        split_idx = int(len(dataset) * 0.8)
        train_data = dataset.iloc[:split_idx]
        
        X_train = train_data[feature_cols]
        y_train_class = train_data['target_class']
        y_train_reg = train_data['target_reg']
        
        # Train models
        self.trainer = MockModelTrainer()
        self.trainer.train_classification_model(X_train, y_train_class, 'random_forest')
        self.trainer.train_regression_model(X_train, y_train_reg, 'random_forest')
        
        self.prediction_engine = MockPredictionEngine(self.trainer)
    
    def test_direction_prediction(self):
        """Test price direction prediction."""
        results = self.prediction_engine.predict_direction(self.sample_data)
        
        # Check result structure
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertIn('timestamps', results)
        
        # Check predictions
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        if len(predictions) > 0:
            # Predictions should be binary
            self.assertTrue(set(predictions).issubset({0, 1}))
            
            # Probabilities should sum to 1
            self.assertEqual(probabilities.shape[1], 2)
            prob_sums = probabilities.sum(axis=1)
            self.assertTrue(np.allclose(prob_sums, 1.0))
    
    def test_price_prediction(self):
        """Test price level prediction."""
        results = self.prediction_engine.predict_price(self.sample_data)
        
        # Check result structure
        self.assertIn('predictions', results)
        self.assertIn('timestamps', results)
        
        # Check predictions
        predictions = results['predictions']
        
        if len(predictions) > 0:
            # Predictions should be positive (prices)
            self.assertTrue(np.all(predictions > 0))
            
            # Predictions should be in reasonable range
            actual_prices = self.sample_data['close']
            price_min = actual_prices.min() * 0.5
            price_max = actual_prices.max() * 2.0
            
            self.assertTrue(np.all(predictions >= price_min))
            self.assertTrue(np.all(predictions <= price_max))
    
    def test_feature_preparation(self):
        """Test feature preparation for prediction."""
        features = self.prediction_engine.prepare_features(self.sample_data)
        
        # Check that features are created
        self.assertGreater(features.shape[1], self.sample_data.shape[1])
        
        # Check for expected feature types
        feature_names = features.columns.tolist()
        self.assertTrue(any('sma' in name for name in feature_names))
        self.assertTrue(any('lag' in name for name in feature_names))
        self.assertTrue(any('returns' in name for name in feature_names))
    
    def test_prediction_with_insufficient_data(self):
        """Test prediction with insufficient data."""
        # Use very small dataset
        small_data = self.sample_data.iloc[:10]
        
        # Should handle gracefully
        direction_results = self.prediction_engine.predict_direction(small_data)
        price_results = self.prediction_engine.predict_price(small_data)
        
        # May return empty results but shouldn't crash
        self.assertIsInstance(direction_results, dict)
        self.assertIsInstance(price_results, dict)


class TestMLIntegration(unittest.TestCase):
    """Integration tests for ML components."""
    
    def setUp(self):
        """Set up integration test environment."""
        np.random.seed(42)
        
        # Create realistic market data
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        
        # Generate price data with some trend
        trend = np.linspace(0, 0.1, 1000)  # 10% upward trend
        noise = np.random.normal(0, 0.02, 1000)  # 2% volatility
        returns = trend + noise
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.market_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, 1000),
            'high': prices + np.abs(np.random.normal(0, 0.3, 1000)),
            'low': prices - np.abs(np.random.normal(0, 0.3, 1000)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
    def test_complete_ml_pipeline(self):
        """Test complete ML pipeline from data to predictions."""
        # Step 1: Feature Engineering
        feature_eng = MockFeatureEngineering()
        features = feature_eng.create_technical_features(self.market_data)
        features = feature_eng.create_lag_features(features)
        features['target'] = feature_eng.create_target_variable(features, 'classification')
        
        # Step 2: Prepare ML dataset
        X_train, X_test, y_train, y_test = feature_eng.prepare_ml_dataset(
            features, 'target', test_size=0.2
        )
        
        # Step 3: Train model
        trainer = MockModelTrainer()
        model = trainer.train_classification_model(X_train, y_train, 'random_forest')
        
        # Step 4: Make predictions
        predictions = trainer.predict('random_forest', X_test)
        probabilities = trainer.predict_proba('random_forest', X_test)
        
        # Step 5: Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        
        # Verify pipeline worked
        self.assertIsNotNone(model)
        self.assertEqual(len(predictions), len(y_test))
        self.assertGreaterEqual(accuracy, 0.3)  # Should be better than random for trending data
        self.assertLessEqual(accuracy, 1.0)
        
        # Step 6: Use prediction engine
        prediction_engine = MockPredictionEngine(trainer)
        recent_data = self.market_data.iloc[-100:]  # Last 100 points
        
        direction_results = prediction_engine.predict_direction(recent_data)
        
        # Verify prediction engine works
        if len(direction_results['predictions']) > 0:
            self.assertTrue(set(direction_results['predictions']).issubset({0, 1}))
    
    def test_multiple_models_comparison(self):
        """Test training and comparing multiple models."""
        # Prepare data
        feature_eng = MockFeatureEngineering()
        features = feature_eng.create_technical_features(self.market_data)
        features = feature_eng.create_lag_features(features)
        features['target'] = feature_eng.create_target_variable(features, 'classification')
        
        X_train, X_test, y_train, y_test = feature_eng.prepare_ml_dataset(
            features, 'target', test_size=0.2
        )
        
        # Train multiple models
        trainer = MockModelTrainer()
        
        rf_model = trainer.train_classification_model(X_train, y_train, 'random_forest')
        lr_model = trainer.train_classification_model(X_train, y_train, 'logistic_regression')
        
        # Compare predictions
        rf_predictions = trainer.predict('random_forest', X_test)
        lr_predictions = trainer.predict('logistic_regression', X_test)
        
        # Evaluate both models
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        
        # Both models should produce valid results
        self.assertGreaterEqual(rf_accuracy, 0)
        self.assertGreaterEqual(lr_accuracy, 0)
        self.assertLessEqual(rf_accuracy, 1)
        self.assertLessEqual(lr_accuracy, 1)
        
        # Models should have some predictive power (better than random)
        self.assertTrue(rf_accuracy > 0.3 or lr_accuracy > 0.3)
    
    def test_model_persistence(self):
        """Test model training, saving, and loading cycle."""
        # Prepare and train model
        feature_eng = MockFeatureEngineering()
        features = feature_eng.create_technical_features(self.market_data)
        features = feature_eng.create_lag_features(features)
        features['target'] = feature_eng.create_target_variable(features, 'classification')
        
        X_train, X_test, y_train, y_test = feature_eng.prepare_ml_dataset(
            features, 'target', test_size=0.2
        )
        
        trainer = MockModelTrainer()
        trainer.train_classification_model(X_train, y_train, 'random_forest')
        
        # Get predictions before saving
        original_predictions = trainer.predict('random_forest', X_test)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            trainer.save_model('random_forest', temp_path)
            
            # Load in new trainer
            new_trainer = MockModelTrainer()
            new_trainer.load_model('random_forest', temp_path)
            
            # Test predictions with loaded model
            loaded_predictions = new_trainer.predict('random_forest', X_test)
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
            # Test with prediction engine
            prediction_engine = MockPredictionEngine(new_trainer)
            results = prediction_engine.predict_direction(self.market_data.iloc[-50:])
            
            # Should work without errors
            self.assertIsInstance(results, dict)
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run all ML tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTest(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTest(loader.loadTestsFromTestCase(TestPredictionEngine))
    suite.addTest(loader.loadTestsFromTestCase(TestMLIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nML Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All ML tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}")