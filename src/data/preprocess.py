"""
Data Preprocessing Module for FOREX Time Series

This module handles data cleaning, feature engineering, and transformation
for FOREX exchange rate data, preparing it for GARCH-LSTM modeling.

Author: Research Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import (
    PREPROCESSING_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR, set_random_seeds
)

warnings.filterwarnings('ignore')


class ForexPreprocessor:
    """
    Preprocess FOREX time series data for modeling.
    
    Features:
    - Handle missing values with multiple strategies
    - Detect and handle outliers
    - Compute log returns and other transformations
    - Feature engineering (technical indicators, rolling statistics)
    - Train/validation/test splitting for time series
    """
    
    def __init__(self, data=None, config=None):
        """
        Initialize preprocessor.
        
        Args:
            data (pd.DataFrame): Raw FOREX data
            config (dict): Preprocessing configuration (uses default if None)
        """
        self.data = data
        self.config = config or PREPROCESSING_CONFIG
        self.processed_data = None
        self.scaler = None
        
        print("Initialized ForexPreprocessor")
        if data is not None:
            print(f"Input data shape: {data.shape}")
    
    def load_data(self, filepath):
        """
        Load data from file.
        
        Args:
            filepath (str or Path): Path to data file (CSV or Parquet)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            self.data = pd.read_csv(filepath)
        elif filepath.suffix == '.parquet':
            self.data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Ensure Datetime column is datetime type
        if 'Datetime' in self.data.columns:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
            self.data.set_index('Datetime', inplace=True)
        
        print(f"✓ Loaded data from {filepath}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        return self.data
    
    def handle_missing_values(self, strategy='forward_fill', max_consecutive=5):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy to handle missing values
                           'forward_fill', 'interpolate', 'drop'
            max_consecutive (int): Maximum consecutive NaNs to fill; drop otherwise
        
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        print(f"\n{'='*60}")
        print("Handling missing values...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        initial_shape = df.shape
        
        # Check missing values
        missing_before = df.isnull().sum()
        print("\nMissing values before handling:")
        for col in missing_before[missing_before > 0].index:
            print(f"  {col}: {missing_before[col]} ({missing_before[col]/len(df)*100:.2f}%)")
        
        # Drop columns with too many missing values (>50%)
        threshold = len(df) * 0.5
        cols_to_drop = missing_before[missing_before > threshold].index.tolist()
        if cols_to_drop:
            print(f"\nDropping columns with >50% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values based on strategy
        if strategy == 'forward_fill':
            # Fill forward (carry last observation forward)
            df = df.fillna(method='ffill', limit=max_consecutive)
            # Fill remaining backward
            df = df.fillna(method='bfill', limit=max_consecutive)
            
        elif strategy == 'interpolate':
            # Linear interpolation
            df = df.interpolate(method='linear', limit=max_consecutive, limit_direction='both')
            
        elif strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Drop rows that still have NaN (beyond max_consecutive)
        df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        
        print(f"\n✓ Missing values handled using '{strategy}'")
        print(f"  Rows before: {initial_shape[0]}")
        print(f"  Rows after: {df.shape[0]}")
        print(f"  Rows dropped: {initial_shape[0] - df.shape[0]}")
        print(f"  Remaining missing values: {missing_after}")
        
        self.data = df
        return df
    
    def detect_outliers(self, columns=None, method='iqr', multiplier=3.0):
        """
        Detect outliers using IQR method.
        
        Args:
            columns (list): Columns to check for outliers (default: price columns)
            method (str): Detection method ('iqr' or 'zscore')
            multiplier (float): IQR multiplier (default: 3.0 for financial data)
        
        Returns:
            pd.DataFrame: Boolean dataframe indicating outliers
        """
        print(f"\n{'='*60}")
        print("Detecting outliers...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close']
            columns = [col for col in columns if col in df.columns]
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                outliers[col] = z_scores > multiplier
        
        # Report outliers
        print(f"Outlier detection method: {method} (multiplier: {multiplier})")
        for col in columns:
            outlier_count = outliers[col].sum()
            if outlier_count > 0:
                pct = (outlier_count / len(df)) * 100
                print(f"  {col}: {outlier_count} outliers ({pct:.2f}%)")
        
        total_outliers = outliers.any(axis=1).sum()
        print(f"\nTotal rows with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)")
        
        return outliers
    
    def compute_log_returns(self, price_column='Close'):
        """
        Compute log returns from price series.
        
        Log returns are used because:
        1. They are time-additive
        2. They are symmetric (easier for modeling)
        3. They approximate percentage changes for small changes
        
        Args:
            price_column (str): Column to compute returns from
        
        Returns:
            pd.Series: Log returns
        """
        print(f"\n{'='*60}")
        print(f"Computing log returns from {price_column}...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        
        if price_column not in df.columns:
            raise ValueError(f"Column {price_column} not found in data")
        
        # Compute log returns: ln(P_t / P_{t-1})
        log_returns = np.log(df[price_column] / df[price_column].shift(1))
        
        # Store in dataframe
        df['Log_Returns'] = log_returns
        
        # Remove first row (NaN from shift)
        df = df.iloc[1:]
        
        # Check for infinite or NaN values
        inf_count = np.isinf(df['Log_Returns']).sum()
        nan_count = df['Log_Returns'].isna().sum()
        
        if inf_count > 0:
            print(f"  Warning: {inf_count} infinite values detected (division by zero)")
            df = df[~np.isinf(df['Log_Returns'])]
        
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values detected")
            df = df[~df['Log_Returns'].isna()]
        
        print(f"✓ Log returns computed")
        print(f"  Mean: {df['Log_Returns'].mean():.6f}")
        print(f"  Std: {df['Log_Returns'].std():.6f}")
        print(f"  Min: {df['Log_Returns'].min():.6f}")
        print(f"  Max: {df['Log_Returns'].max():.6f}")
        print(f"  Skewness: {df['Log_Returns'].skew():.4f}")
        print(f"  Kurtosis: {df['Log_Returns'].kurtosis():.4f}")
        
        self.data = df
        return df['Log_Returns']
    
    def compute_log_trading_range(self):
        """
        Compute logarithmic trading range: ln(High) - ln(Low).
        This measures intraday volatility.
        """
        print(f"\n{'='*60}")
        print("Computing log trading range...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        
        if 'High' not in df.columns or 'Low' not in df.columns:
            print("  Skipping: High/Low columns not available")
            return None
        
        df['Log_Trading_Range'] = np.log(df['High']) - np.log(df['Low'])
        
        print(f"✓ Log trading range computed")
        print(f"  Mean: {df['Log_Trading_Range'].mean():.6f}")
        print(f"  Std: {df['Log_Trading_Range'].std():.6f}")
        
        self.data = df
        return df['Log_Trading_Range']
    
    def compute_rolling_volatility(self, windows=[10, 30, 60]):
        """
        Compute rolling volatility (standard deviation of returns).
        
        Args:
            windows (list): List of window sizes in days
        
        Returns:
            pd.DataFrame: Data with rolling volatility columns
        """
        print(f"\n{'='*60}")
        print("Computing rolling volatility...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        
        if 'Log_Returns' not in df.columns:
            raise ValueError("Log_Returns not found. Call compute_log_returns() first.")
        
        for window in windows:
            col_name = f'Volatility_{window}D'
            df[col_name] = df['Log_Returns'].rolling(window=window).std()
            
            # Drop NaN rows from rolling calculation
            valid_count = df[col_name].notna().sum()
            print(f"  {col_name}: {valid_count} valid values")
        
        print(f"✓ Rolling volatility computed for windows: {windows}")
        
        self.data = df
        return df
    
    def compute_technical_indicators(self):
        """
        Compute technical indicators (RSI, SMA, EMA, MACD).
        These capture momentum and trend information.
        """
        print(f"\n{'='*60}")
        print("Computing technical indicators...")
        print(f"{'='*60}")
        
        df = self.data.copy()
        
        if 'Close' not in df.columns:
            print("  Skipping: Close price not available")
            return df
        
        close = df['Close']
        
        # Relative Strength Index (RSI)
        period = self.config['technical_indicator_periods']['rsi']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        print(f"  ✓ RSI_{period}")
        
        # Simple Moving Averages (SMA)
        for period in self.config['technical_indicator_periods']['sma']:
            df[f'SMA_{period}'] = close.rolling(window=period).mean()
            print(f"  ✓ SMA_{period}")
        
        # Exponential Moving Averages (EMA)
        for period in self.config['technical_indicator_periods']['ema']:
            df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            print(f"  ✓ EMA_{period}")
        
        # MACD (Moving Average Convergence Divergence)
        fast = self.config['technical_indicator_periods']['macd_fast']
        slow = self.config['technical_indicator_periods']['macd_slow']
        signal = self.config['technical_indicator_periods']['macd_signal']
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        print(f"  ✓ MACD ({fast}, {slow}, {signal})")
        
        print(f"✓ Technical indicators computed")
        
        self.data = df
        return df
    
    def split_data(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """
        Split time series data into train, validation, and test sets.
        
        Important: For time series, we split chronologically (no random shuffle).
        Most recent data is used for testing to simulate real forecasting.
        
        Args:
            train_ratio (float): Proportion for training
            val_ratio (float): Proportion for validation
            test_ratio (float): Proportion for testing
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print(f"\n{'='*60}")
        print("Splitting data (chronological split for time series)...")
        print(f"{'='*60}")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        df = self.data.copy()
        n = len(df)
        
        # Calculate split indices
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split data
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"Dataset splitting:")
        print(f"  Total samples: {n}")
        print(f"\n  Training set:")
        print(f"    Samples: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"    Date range: {train_df.index.min()} to {train_df.index.max()}")
        print(f"\n  Validation set:")
        print(f"    Samples: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"    Date range: {val_df.index.min()} to {val_df.index.max()}")
        print(f"\n  Test set:")
        print(f"    Samples: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        print(f"    Date range: {test_df.index.min()} to {test_df.index.max()}")
        
        return train_df, val_df, test_df
    
    def preprocess_pipeline(self, save_output=True):
        """
        Run full preprocessing pipeline.
        
        Args:
            save_output (bool): Whether to save processed data
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print("\n" + "="*80)
        print("RUNNING FULL PREPROCESSING PIPELINE")
        print("="*80)
        
        # 1. Handle missing values
        self.handle_missing_values(
            strategy=self.config['missing_value_strategy'],
            max_consecutive=self.config['max_missing_consecutive']
        )
        
        # 2. Detect outliers (just report, don't remove yet for financial data)
        if self.config['outlier_detection']:
            outliers = self.detect_outliers(
                multiplier=self.config['outlier_iqr_multiplier']
            )
        
        # 3. Compute log returns
        if self.config['apply_log_returns']:
            self.compute_log_returns()
        
        # 4. Compute log trading range
        self.compute_log_trading_range()
        
        # 5. Compute rolling volatility
        self.compute_rolling_volatility(
            windows=self.config['rolling_volatility_windows']
        )
        
        # 6. Compute technical indicators
        self.compute_technical_indicators()
        
        # 7. Drop remaining NaN rows (from rolling calculations)
        initial_len = len(self.data)
        self.data = self.data.dropna()
        dropped = initial_len - len(self.data)
        print(f"\n✓ Dropped {dropped} rows with NaN from rolling calculations")
        
        # 8. Split data
        train_df, val_df, test_df = self.split_data(
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['validation_ratio'],
            test_ratio=self.config['test_ratio']
        )
        
        # 9. Save processed data
        if save_output:
            print(f"\n{'='*60}")
            print("Saving processed data...")
            print(f"{'='*60}")
            
            train_df.to_csv(PROCESSED_DATA_DIR / 'train_data.csv')
            val_df.to_csv(PROCESSED_DATA_DIR / 'val_data.csv')
            test_df.to_csv(PROCESSED_DATA_DIR / 'test_data.csv')
            
            print(f"✓ Saved: {PROCESSED_DATA_DIR / 'train_data.csv'}")
            print(f"✓ Saved: {PROCESSED_DATA_DIR / 'val_data.csv'}")
            print(f"✓ Saved: {PROCESSED_DATA_DIR / 'test_data.csv'}")
        
        print("\n" + "="*80)
        print("✓ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return train_df, val_df, test_df


def main():
    """
    Main function to run preprocessing.
    """
    # Set random seeds
    set_random_seeds()
    
    print("\n" + "="*80)
    print("FOREX DATA PREPROCESSING MODULE")
    print("="*80)
    
    # Find the most recent raw data file
    raw_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("✗ No raw data files found. Run fetch_data.py first.")
        return
    
    # Use the most recent file
    latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLoading data from: {latest_file.name}")
    
    # Initialize preprocessor
    preprocessor = ForexPreprocessor()
    preprocessor.load_data(latest_file)
    
    # Run preprocessing pipeline
    train, val, test = preprocessor.preprocess_pipeline(save_output=True)
    
    # Display final statistics
    print("\nFinal processed data summary:")
    print(f"  Training features: {train.shape[1]}")
    print(f"  Features: {list(train.columns)}")
    
    return train, val, test


if __name__ == "__main__":
    train, val, test = main()
