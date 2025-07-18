"""
Data processing utilities for the Quotex trading bot.
Handles data cleaning, transformation, normalization, and preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

from .logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Main data processing class for trading data."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize DataProcessor.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler(scaler_type)
        self.imputer = None
        self.is_fitted = False
        
    def _get_scaler(self, scaler_type: str):
        """Get the specified scaler."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(scaler_type, StandardScaler())
    
    def clean_ohlc_data(self, df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """
        Clean OHLC (Open, High, Low, Close) trading data.
        
        Args:
            df: DataFrame with OHLC data
            validate: Whether to validate OHLC relationships
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning OHLC data")
        df_clean = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        for col in required_cols + ['volume'] if 'volume' in df_clean.columns else required_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with all NaN OHLC values
        df_clean = df_clean.dropna(subset=required_cols, how='all')
        
        # Validate OHLC relationships
        if validate:
            df_clean = self._validate_ohlc_relationships(df_clean)
        
        # Handle extreme outliers
        df_clean = self._handle_price_outliers(df_clean)
        
        # Ensure datetime index
        df_clean = self._ensure_datetime_index(df_clean)
        
        # Sort by timestamp
        df_clean = df_clean.sort_index()
        
        logger.info(f"Cleaned OHLC data: {len(df_clean)} rows remaining")
        return df_clean
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLC price relationships."""
        logger.debug("Validating OHLC relationships")
        
        # Check if high >= max(open, close) and low <= min(open, close)
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        
        if invalid_high.any():
            logger.warning(f"Found {invalid_high.sum()} rows with invalid high prices")
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
        
        if invalid_low.any():
            logger.warning(f"Found {invalid_low.sum()} rows with invalid low prices")
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
        
        return df
    
    def _handle_price_outliers(self, df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """Handle extreme price outliers using Z-score method."""
        logger.debug("Handling price outliers")
        
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = z_scores > z_threshold
                
                if outliers.any():
                    logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    # Replace outliers with median
                    median_val = df[col].median()
                    df.loc[df[col].index[outliers], col] = median_val
        
        return df
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
                df = df.drop('timestamp', axis=1)
            elif 'time' in df.columns:
                df.index = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
            else:
                logger.warning("No timestamp column found, using sequential index")
        
        return df
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLC data to different timeframe.
        
        Args:
            df: DataFrame with OHLC data
            timeframe: Target timeframe ('1min', '5min', '15min', '1H', '4H', '1D')
        
        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {timeframe}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for resampling")
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        # Add volume if present
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        
        # Add other numeric columns with mean aggregation
        for col in df.columns:
            if col not in agg_dict and pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
        
        resampled = df.resample(timeframe).agg(agg_dict).dropna()
        
        logger.info(f"Resampled data: {len(resampled)} rows")
        return resampled
    
    def add_returns(self, df: pd.DataFrame, periods: List[int] = [1]) -> pd.DataFrame:
        """
        Add return columns to the DataFrame.
        
        Args:
            df: DataFrame with price data
            periods: List of periods for return calculation
        
        Returns:
            DataFrame with return columns
        """
        logger.debug(f"Adding returns for periods: {periods}")
        df_returns = df.copy()
        
        for period in periods:
            # Simple returns
            df_returns[f'return_{period}'] = df_returns['close'].pct_change(period)
            
            # Log returns
            df_returns[f'log_return_{period}'] = np.log(df_returns['close'] / df_returns['close'].shift(period))
        
        return df_returns
    
    def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add volatility measures to the DataFrame.
        
        Args:
            df: DataFrame with return data
            window: Rolling window for volatility calculation
        
        Returns:
            DataFrame with volatility columns
        """
        logger.debug(f"Adding volatility with window: {window}")
        df_vol = df.copy()
        
        # True Range
        df_vol['tr'] = np.maximum(
            df_vol['high'] - df_vol['low'],
            np.maximum(
                np.abs(df_vol['high'] - df_vol['close'].shift(1)),
                np.abs(df_vol['low'] - df_vol['close'].shift(1))
            )
        )
        
        # Average True Range
        df_vol['atr'] = df_vol['tr'].rolling(window=window).mean()
        
        # Price volatility (standard deviation of returns)
        if 'return_1' in df_vol.columns:
            df_vol['volatility'] = df_vol['return_1'].rolling(window=window).std()
        
        # Realized volatility (sum of squared returns)
        if 'return_1' in df_vol.columns:
            df_vol['realized_vol'] = np.sqrt(
                df_vol['return_1'].rolling(window=window).apply(lambda x: (x**2).sum())
            )
        
        return df_vol
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = "forward_fill", 
                          max_gap: Optional[int] = None) -> pd.DataFrame:
        """
        Handle missing data in the DataFrame.
        
        Args:
            df: DataFrame with missing data
            method: Method to handle missing data ('forward_fill', 'backward_fill', 'interpolate', 'knn')
            max_gap: Maximum gap size to fill
        
        Returns:
            DataFrame with missing data handled
        """
        logger.info(f"Handling missing data using method: {method}")
        df_filled = df.copy()
        
        # Log missing data info
        missing_counts = df_filled.isnull().sum()
        if missing_counts.any():
            logger.info(f"Missing data counts: {missing_counts[missing_counts > 0].to_dict()}")
        
        if method == "forward_fill":
            if max_gap:
                df_filled = df_filled.fillna(method='ffill', limit=max_gap)
            else:
                df_filled = df_filled.fillna(method='ffill')
                
        elif method == "backward_fill":
            if max_gap:
                df_filled = df_filled.fillna(method='bfill', limit=max_gap)
            else:
                df_filled = df_filled.fillna(method='bfill')
                
        elif method == "interpolate":
            df_filled = df_filled.interpolate(method='time' if isinstance(df_filled.index, pd.DatetimeIndex) else 'linear')
            
        elif method == "knn":
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        
        # Final check and remove rows with remaining NaN values
        remaining_na = df_filled.isnull().sum().sum()
        if remaining_na > 0:
            logger.warning(f"Removing {remaining_na} remaining NaN values")
            df_filled = df_filled.dropna()
        
        return df_filled
    
    def normalize_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          fit: bool = True) -> pd.DataFrame:
        """
        Normalize/scale feature columns.
        
        Args:
            df: DataFrame to normalize
            columns: Columns to normalize (if None, all numeric columns)
            fit: Whether to fit the scaler
        
        Returns:
            DataFrame with normalized features
        """
        logger.debug(f"Normalizing features using {self.scaler_type} scaler")
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            df_norm[columns] = self.scaler.fit_transform(df_norm[columns])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform")
            df_norm[columns] = self.scaler.transform(df_norm[columns])
        
        return df_norm
    
    def inverse_normalize(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse normalize features back to original scale.
        
        Args:
            df: Normalized DataFrame
            columns: Columns to inverse normalize
        
        Returns:
            DataFrame with original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        df_inv = df.copy()
        
        if columns is None:
            columns = df_inv.select_dtypes(include=[np.number]).columns.tolist()
        
        df_inv[columns] = self.scaler.inverse_transform(df_inv[columns])
        
        return df_inv
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from price data.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating additional features")
        df_features = df.copy()
        
        # Price-based features
        df_features['hl_avg'] = (df_features['high'] + df_features['low']) / 2
        df_features['hlc_avg'] = (df_features['high'] + df_features['low'] + df_features['close']) / 3
        df_features['ohlc_avg'] = (df_features['open'] + df_features['high'] + 
                                  df_features['low'] + df_features['close']) / 4
        
        # Price ranges
        df_features['hl_range'] = df_features['high'] - df_features['low']
        df_features['oc_range'] = abs(df_features['open'] - df_features['close'])
        
        # Gap features
        df_features['gap'] = df_features['open'] - df_features['close'].shift(1)
        df_features['gap_pct'] = df_features['gap'] / df_features['close'].shift(1)
        
        # Time-based features (if datetime index)
        if isinstance(df_features.index, pd.DatetimeIndex):
            df_features['hour'] = df_features.index.hour
            df_features['day_of_week'] = df_features.index.dayofweek
            df_features['month'] = df_features.index.month
            df_features['quarter'] = df_features.index.quarter
        
        # Volume features (if volume column exists)
        if 'volume' in df_features.columns:
            df_features['volume_sma'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            df_features['price_volume'] = df_features['close'] * df_features['volume']
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def split_sequences(self, data: np.ndarray, n_steps_in: int, n_steps_out: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a multivariate sequence into samples for ML models.
        
        Args:
            data: Input data array
            n_steps_in: Number of time steps for input
            n_steps_out: Number of time steps for output
        
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data)):
            # Find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            
            # Check if we are beyond the dataset
            if out_end_ix > len(data):
                break
                
            # Gather input and output parts of the pattern
            seq_x = data[i:end_ix]
            seq_y = data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        
        return np.array(X), np.array(y)


def detect_market_regime(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Detect market regime (trending vs ranging).
    
    Args:
        df: DataFrame with price data
        window: Window for regime detection
    
    Returns:
        DataFrame with regime indicators
    """
    logger.debug(f"Detecting market regime with window: {window}")
    df_regime = df.copy()
    
    # Calculate price efficiency ratio
    price_change = abs(df_regime['close'] - df_regime['close'].shift(window))
    price_volatility = df_regime['close'].diff().abs().rolling(window).sum()
    
    df_regime['efficiency_ratio'] = price_change / price_volatility
    
    # Classify regime
    df_regime['market_regime'] = np.where(
        df_regime['efficiency_ratio'] > 0.3, 'trending', 'ranging'
    )
    
    return df_regime


def calculate_fractal_dimension(series: pd.Series, max_k: int = 10) -> float:
    """
    Calculate fractal dimension of a time series.
    
    Args:
        series: Time series data
        max_k: Maximum k value for calculation
    
    Returns:
        Fractal dimension
    """
    n = len(series)
    rs_values = []
    
    for k in range(2, min(max_k + 1, n // 2)):
        # Split series into k subseries
        subseries_length = n // k
        rs_list = []
        
        for i in range(k):
            start_idx = i * subseries_length
            end_idx = start_idx + subseries_length
            subseries = series.iloc[start_idx:end_idx]
            
            # Calculate R/S statistic
            mean_val = subseries.mean()
            deviations = subseries - mean_val
            cumulative_deviations = deviations.cumsum()
            
            R = cumulative_deviations.max() - cumulative_deviations.min()
            S = subseries.std()
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return 0.5  # Random walk
    
    # Calculate Hurst exponent
    log_rs = np.log(rs_values)
    log_n = np.log(range(2, len(rs_values) + 2))
    
    hurst_exponent = np.polyfit(log_n, log_rs, 1)[0]
    
    # Convert to fractal dimension
    fractal_dimension = 2 - hurst_exponent
    
    return fractal_dimension


# Utility functions
def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return report.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Data quality report
    """
    logger.info("Validating data quality")
    
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'date_range': None
    }
    
    # Date range info
    if isinstance(df.index, pd.DatetimeIndex):
        report['date_range'] = {
            'start': df.index.min().isoformat(),
            'end': df.index.max().isoformat(),
            'frequency': pd.infer_freq(df.index)
        }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    logger.info(f"Data quality report generated for {len(df)} rows")
    return report


if __name__ == "__main__":
    # Example usage
    logger = get_logger("data_processor_test")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate sample OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    
    sample_data = pd.DataFrame({
        'open': close_prices + np.random.randn(1000) * 0.1,
        'high': close_prices + np.abs(np.random.randn(1000) * 0.2),
        'low': close_prices - np.abs(np.random.randn(1000) * 0.2),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Test data processor
    processor = DataProcessor()
    
    # Clean data
    clean_data = processor.clean_ohlc_data(sample_data)
    logger.info(f"Original data: {len(sample_data)} rows")
    logger.info(f"Clean data: {len(clean_data)} rows")
    
    # Add features
    feature_data = processor.create_features(clean_data)
    logger.info(f"Features added: {len(feature_data.columns) - len(clean_data.columns)}")
    
    # Validate data quality
    quality_report = validate_data_quality(feature_data)
    logger.info(f"Data quality report: {quality_report['total_rows']} rows, "
               f"{quality_report['total_columns']} columns")
    
    print("Data processor test completed successfully!")